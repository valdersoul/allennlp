import logging
import numpy as np
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.functional import nll_loss

from allennlp.common import Params, squad_eval
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Highway
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.models.tri_linear_att import TriLinearAttention
#from allennlp.modules.variational_dropout import VariationalDropout
from allennlp.nn import InitializerApplicator, util
from allennlp.training.metrics import Average, BooleanAccuracy, CategoricalAccuracy

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("bidaf-self-atten")
class BidafPlusSelfAttention(Model):

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 phrase_layer: Seq2SeqEncoder,
                 summary_layer: Seq2SeqEncoder,
                 residual_encoder: Seq2SeqEncoder,
                 span_start_encoder: Seq2SeqEncoder,
                 span_end_encoder: Seq2SeqEncoder,
                 initializer: InitializerApplicator,
                 dropout: float = 0.2,
                 mask_lstms: bool = True) -> None:
        super(BidafPlusSelfAttention, self).__init__(vocab)

        self._text_field_embedder = text_field_embedder
        self._phrase_layer = phrase_layer

        self._summary_layer = summary_layer
        self._fcx = TimeDistributed(torch.nn.Linear(200, 200))
        self._fch = TimeDistributed(torch.nn.Linear(200, 200))

        self._matrix_attention = TriLinearAttention(200)
        self._matrix_attention_ruminate = TriLinearAttention(200)
        self._highway_layer = TimeDistributed(Highway(text_field_embedder.get_output_dim(),
                                        2))
        self._merge_atten = TimeDistributed(torch.nn.Linear(200 * 4, 200))
        self._merge_gate = TimeDistributed(torch.nn.Linear(200 * 4, 200))

        self._residual_encoder = residual_encoder
        self._self_atten = TriLinearAttention(200)
        self._merge_self_atten = TimeDistributed(torch.nn.Linear(200 * 3, 200))
        self._merge_question = TimeDistributed(torch.nn.Linear(200 * 3, 200))
        self._merge_question_gate = TimeDistributed(torch.nn.Linear(200 * 3, 200))
        self._question_attention = TriLinearAttention(200)

        self._span_start_encoder = span_start_encoder
        self._span_end_encoder = span_end_encoder

        self._span_start_predictor = TimeDistributed(torch.nn.Linear(200, 1))
        self._span_end_predictor = TimeDistributed(torch.nn.Linear(200, 1))

        initializer(self)

        self._span_start_accuracy = CategoricalAccuracy()
        self._span_end_accuracy = CategoricalAccuracy()
        self._span_accuracy = BooleanAccuracy()
        self._official_em = Average()
        self._official_f1 = Average()
        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
            #self._dropout = VariationalDropout(p=dropout)
        else:
            raise ValueError()
            # self._dropout = lambda x: x
        self._mask_lstms = mask_lstms

        # RUMINATE LAYER START
        self._merge_atten_ruminate = TimeDistributed(torch.nn.Linear(200 * 4, 200))
        self._merge_gate_ruminate = TimeDistributed(torch.nn.Linear(200 * 4, 200))
        self._fcqz = TimeDistributed(torch.nn.Linear(200, 200))
        self._fcqx = TimeDistributed(torch.nn.Linear(200, 200))
        self._fcfqz = TimeDistributed(torch.nn.Linear(200, 200))
        self._fcfqx = TimeDistributed(torch.nn.Linear(200, 200))

        self._fccz = TimeDistributed(torch.nn.Linear(200, 200))
        self._fccx = TimeDistributed(torch.nn.Linear(200, 200))
        self._fcfcz = TimeDistributed(torch.nn.Linear(200, 200))
        self._fcfcx = TimeDistributed(torch.nn.Linear(200, 200))
        # RUMINATE LAYER END        

        #Query type embedding
        self._query_type_embedding = torch.nn.Embedding(11, 20)
        self._query_projection = torch.nn.Linear(20, 271)

    def forward(self,  # type: ignore
                question: Dict[str, torch.LongTensor],
                question_type: Dict[str, torch.LongTensor],
                passage: Dict[str, torch.LongTensor],
                passage_em: Dict[str, torch.LongTensor],
                question_em: Dict[str, torch.LongTensor],
                sentences: Dict[str, torch.LongTensor],
                question_sentence: Dict[str, torch.LongTensor],
                correct_sentence: Dict[str, torch.LongTensor],
                correct_label: Dict[str, torch.LongTensor],
                span_start: torch.IntTensor = None,
                span_end: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        question : Dict[str, torch.LongTensor]
            From a ``TextField``.
        passage : Dict[str, torch.LongTensor]
            From a ``TextField``.  The model assumes that this passage contains the answer to the
            question, and predicts the beginning and ending positions of the answer within the
            passage.
        span_start : ``torch.IntTensor``, optional
            From an ``IndexField``.  This is one of the things we are trying to predict - the
            beginning position of the answer with the passage.  This is an `inclusive` index.  If
            this is given, we will compute a loss that gets included in the output dictionary.
        span_end : ``torch.IntTensor``, optional
            From an ``IndexField``.  This is one of the things we are trying to predict - the
            ending position of the answer with the passage.  This is an `inclusive` index.  If
            this is given, we will compute a loss that gets included in the output dictionary.
        metadata : ``List[Dict[str, Any]]``, optional
            If present, this should contain the question ID, original passage text, and token
            offsets into the passage for each instance in the batch.  We use this for computing
            official metrics using the official SQuAD evaluation script.  The length of this list
            should be the batch size, and each dictionary should have the keys ``id``,
            ``original_passage``, and ``token_offsets``.  If you only want the best span string and
            don't care about official metrics, you can omit the ``id`` key.
        Returns
        -------
        An output dictionary consisting of:
        span_start_logits : torch.FloatTensor
            A tensor of shape ``(batch_size, passage_length)`` representing unnormalised log
            probabilities of the span start position.
        span_start_probs : torch.FloatTensor
            The result of ``softmax(span_start_logits)``.
        span_end_logits : torch.FloatTensor
            A tensor of shape ``(batch_size, passage_length)`` representing unnormalised log
            probabilities of the span end position (inclusive).
        span_end_probs : torch.FloatTensor
            The result of ``softmax(span_end_logits)``.
        best_span : torch.IntTensor
            The result of a constrained inference over ``span_start_logits`` and
            ``span_end_logits`` to find the most probable span.  Shape is ``(batch_size, 2)``.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        best_span_str : List[str]
            If sufficient metadata was provided for the instances in the batch, we also return the
            string from the original passage that the model thinks is the best answer to the
            question.
        """
        question_type_embedding = self._query_type_embedding(question_type)
        question_type_embedding = self._query_projection(question_type_embedding)

        question_embedding, question_extra = self._text_field_embedder(question)
        passage_embedding, passage_extra = self._text_field_embedder(passage)
        embedded_question = self._dropout(self._highway_layer(question_embedding))
        embedded_passage = self._dropout(self._highway_layer(passage_embedding))

        batch_size = embedded_question.size(0)
        passage_length = embedded_passage.size(1)
        question_length = embedded_question.size(1)
        question_mask = util.get_text_field_mask(question).float()
        passage_mask = util.get_text_field_mask(passage).float()
        question_lstm_mask = question_mask if self._mask_lstms else None
        passage_lstm_mask = passage_mask if self._mask_lstms else None

        embedded_passage = torch.cat([embedded_passage, passage_extra, passage_em.float().unsqueeze(-1)], dim=-1)
        embedded_question = torch.cat([embedded_question, question_extra, question_em.float().unsqueeze(-1)], dim=-1) + question_type_embedding

        encoded_question = self._dropout(self._phrase_layer(embedded_question, question_lstm_mask))
        encoded_passage = self._dropout(self._phrase_layer(embedded_passage, passage_lstm_mask))
        encoding_dim = encoded_question.size(-1)

        # Shape: (batch_size, passage_length, question_length)
        passage_question_similarity = self._matrix_attention(encoded_passage, encoded_question)
        # Shape: (batch_size, passage_length, question_length)
        passage_question_attention = util.last_dim_softmax(passage_question_similarity, question_mask)
        # Shape: (batch_size, passage_length, encoding_dim)
        passage_question_vectors = util.weighted_sum(encoded_question, passage_question_attention)

        # We replace masked values with something really negative here, so they don't affect the
        # max below.
        masked_similarity = util.replace_masked_values(passage_question_similarity,
                                                       question_mask.unsqueeze(1),
                                                       -1e7)
        # Shape: (batch_size, passage_length)
        question_passage_similarity = masked_similarity.max(dim=-1)[0].squeeze(-1)
        # Shape: (batch_size, passage_length)
        question_passage_attention = util.masked_softmax(question_passage_similarity, passage_mask)
        # Shape: (batch_size, encoding_dim)
        question_passage_vector = util.weighted_sum(encoded_passage, question_passage_attention)
        # Shape: (batch_size, passage_length, encoding_dim)
        tiled_question_passage_vector = question_passage_vector.unsqueeze(1).expand(batch_size,
                                                                                    passage_length,
                                                                                    encoding_dim)

        # Shape: (batch_size, passage_length, encoding_dim * 4)
        final_merged_passage = torch.cat([encoded_passage,
                                          passage_question_vectors,
                                          encoded_passage * passage_question_vectors,
                                          encoded_passage * tiled_question_passage_vector],
                                         dim=-1)

        final_merged_passage_ = F.tanh(self._merge_atten(final_merged_passage))
        final_gate = F.sigmoid(self._merge_gate(final_merged_passage))
        final_merged_passage = final_merged_passage_ * final_gate

        # RUMINATE LAYER START
        summary = self._dropout(self._summary_layer(self._dropout(final_merged_passage), passage_mask))
        x = self._fcx(final_merged_passage)
        h = self._fch(summary)
        a = torch.mean(x * F.sigmoid(x + h), 1).unsqueeze(1) 

        ##Query
        q = a.repeat(1,question_length,1)
        zq = F.tanh(self._fcqx(encoded_question) + self._fcqz(q))
        fq = F.sigmoid(self._fcfqx(encoded_question) + self._fcfqz(q))
        encoded_question = fq * encoded_question + (1 - fq) * zq

        ##Context
        c = a.repeat(1,passage_length,1)
        zc = F.tanh(self._fccx(encoded_passage) + self._fccz(c))
        fc = F.sigmoid(self._fcfcx(encoded_passage) + self._fcfcz(c))
        encoded_passage = fc * encoded_passage + (1 - fc) * zc

        # Shape: (batch_size, passage_length, question_length)
        passage_question_similarity = self._matrix_attention_ruminate(encoded_passage, encoded_question)
        # Shape: (batch_size, passage_length, question_length)
        passage_question_attention = util.last_dim_softmax(passage_question_similarity, question_mask)
        # Shape: (batch_size, passage_length, encoding_dim)
        passage_question_vectors = util.weighted_sum(encoded_question, passage_question_attention)

        # We replace masked values with something really negative here, so they don't affect the
        # max below.
        masked_similarity = util.replace_masked_values(passage_question_similarity,
                                                       question_mask.unsqueeze(1),
                                                       -1e7)
        # Shape: (batch_size, passage_length)
        question_passage_similarity = masked_similarity.max(dim=-1)[0].squeeze(-1)
        # Shape: (batch_size, passage_length)
        question_passage_attention = util.masked_softmax(question_passage_similarity, passage_mask)
        # Shape: (batch_size, encoding_dim)
        question_passage_vector = util.weighted_sum(encoded_passage, question_passage_attention)
        # Shape: (batch_size, passage_length, encoding_dim)
        tiled_question_passage_vector = question_passage_vector.unsqueeze(1).expand(batch_size,
                                                                                    passage_length,
                                                                                    encoding_dim)

        # Shape: (batch_size, passage_length, encoding_dim * 4)
        final_merged_passage = torch.cat([encoded_passage,
                                          passage_question_vectors,
                                          encoded_passage * passage_question_vectors,
                                          encoded_passage * tiled_question_passage_vector],
                                         dim=-1)

        final_merged_passage_ = F.tanh(self._merge_atten_ruminate(final_merged_passage))
        final_gate = F.sigmoid(self._merge_gate_ruminate(final_merged_passage))
        final_merged_passage = final_merged_passage_ * final_gate
        # RUMINATE LAYER END


        residual_layer = self._dropout(self._residual_encoder(self._dropout(final_merged_passage), passage_mask))
        self_atten_matrix = self._self_atten(residual_layer, residual_layer)
        mask = passage_mask.resize(batch_size, passage_length, 1) * passage_mask.resize(batch_size, 1, passage_length)

        # torch.eye does not have a gpu implementation, so we are forced to use the cpu one and .cuda()
        # Not sure if this matters for performance
        position_matrix = np.repeat(np.resize(np.arange(passage_length), [1, passage_length]), [passage_length], axis=0)
        position_matrix = Variable(torch.from_numpy(np.power(0.99,abs(np.transpose(position_matrix)-position_matrix))).float()).cuda()

        self_mask = Variable(torch.eye(passage_length, passage_length).cuda()).resize(1, passage_length, passage_length)
        mask = mask * (1 - self_mask) * position_matrix
        self_atten_probs = util.last_dim_softmax(self_atten_matrix, mask)

        # Batch matrix multiplication:
        # (batch, passage_len, passage_len) * (batch, passage_len, dim) -> (batch, passage_len, dim)
        self_atten_vecs = torch.matmul(self_atten_probs, residual_layer)

        residual_layer = F.tanh(self._merge_self_atten(torch.cat(
            [self_atten_vecs, residual_layer, residual_layer * self_atten_vecs], dim=-1)))

        final_merged_passage += residual_layer

        # passage_question_similarity = self._question_attention(final_merged_passage, encoded_question)
        # passage_question_attention = util.last_dim_softmax(passage_question_similarity, question_mask)
        # passage_question_vectors = util.weighted_sum(encoded_question, passage_question_attention)
        
        avg_question = torch.mean(encoded_question,1).unsqueeze(1).repeat(1, passage_length, 1)

        final_merged_passage = torch.cat(
            [final_merged_passage, avg_question, final_merged_passage * avg_question], dim=-1
        )

        final_merged_passage_ = F.tanh(self._merge_question(final_merged_passage))
        final_gate = F.sigmoid(self._merge_question_gate(final_merged_passage))

        final_merged_passage = final_gate * final_merged_passage_
        final_merged_passage = self._dropout(final_merged_passage)
        

        start_rep = self._span_start_encoder(final_merged_passage, passage_lstm_mask)
        span_start_logits = self._span_start_predictor(start_rep).squeeze(-1)
        span_start_probs = util.masked_softmax(span_start_logits, passage_mask)

        end_rep = self._span_end_encoder(torch.cat([final_merged_passage, start_rep], dim=-1), passage_lstm_mask)
        span_end_logits = self._span_end_predictor(end_rep).squeeze(-1)
        span_end_probs = util.masked_softmax(span_end_logits, passage_mask)

        span_start_logits = util.replace_masked_values(span_start_logits, passage_mask, 0)
        span_end_logits = util.replace_masked_values(span_end_logits, passage_mask, 0)

        best_span = self._get_best_span(span_start_probs, span_end_probs)


        output_dict = {"span_start_logits": span_start_logits,
                       "span_start_probs": span_start_probs,
                       "span_end_logits": span_end_logits,
                       "span_end_probs": span_end_probs,
                       "best_span": best_span}
        if span_start is not None:
            loss = nll_loss(util.masked_log_softmax(span_start_logits, passage_mask), span_start.squeeze(-1))
            self._span_start_accuracy(span_start_logits, span_start.squeeze(-1))
            loss += nll_loss(util.masked_log_softmax(span_end_logits, passage_mask), span_end.squeeze(-1))
            self._span_end_accuracy(span_end_logits, span_end.squeeze(-1))
            self._span_accuracy(best_span[:,:2].contiguous(), torch.stack([span_start, span_end], -1))
            output_dict["loss"] = loss
        if metadata is not None:
            output_dict['best_span_str'] = []
            output_dict['predict'] = {}
            output_dict['predicts'] = {}
            for i in range(batch_size):
                passage_str = metadata[i]['original_passage']
                offsets = metadata[i]['token_offsets']
                predicted_span = tuple(best_span[i].data.cpu().numpy())
                start_offset = offsets[predicted_span[0]][0]
                end_offset = offsets[predicted_span[1]][1]

                candidate_answers=[]
                candidate_num = int(len(predicted_span) / 2)
                for j in range(candidate_num - 1):
                    s_offset = offsets[predicted_span[2 * (j + 1)]][0]
                    e_offset = offsets[predicted_span[2 * (j + 1) + 1]][1]
                    string = passage_str[s_offset:e_offset]
                    candidate_answers.append(string)

                best_span_string = passage_str[start_offset:end_offset]
                candidate_answers.append(best_span_string)
                output_dict['best_span_str'].append(best_span_string)
                answer_texts = metadata[i].get('answer_texts', [])
                question_text = metadata[i].get('question', [])
                output_dict['predict'][metadata[i].get('question_id', [])] = best_span_string
                if best_span_string not in answer_texts:
                    output_dict['predicts'][metadata[i].get('question_id', [])] = [passage_str, question_text, candidate_answers, answer_texts]
                exact_match = f1_score = 0
                if answer_texts:
                    exact_match = squad_eval.metric_max_over_ground_truths(
                            squad_eval.exact_match_score,
                            best_span_string,
                            answer_texts)
                    f1_score = squad_eval.metric_max_over_ground_truths(
                            squad_eval.f1_score,
                            best_span_string,
                            answer_texts)
                self._official_em(100 * exact_match)
                self._official_f1(100 * f1_score)
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
                'start_acc': self._span_start_accuracy.get_metric(reset),
                'end_acc': self._span_end_accuracy.get_metric(reset),
                'span_acc': self._span_accuracy.get_metric(reset),
                'em': self._official_em.get_metric(reset),
                'f1': self._official_f1.get_metric(reset),
                }

    @staticmethod
    def _get_best_span(span_start_logits: Variable, span_end_logits: Variable) -> Variable:
        if span_start_logits.dim() != 2 or span_end_logits.dim() != 2:
            raise ValueError("Input shapes must be (batch_size, passage_length)")
        batch_size, passage_length = span_start_logits.size()
        max_span_log_prob = [-1e20] * batch_size
        span_start_argmax = [0] * batch_size

        max_answer_span = 17
        candidate_num = 5

        best_word_span = Variable(span_start_logits.data.new()
                            .resize_(batch_size, 2*candidate_num).fill_(0)).long()

        start_end_matrix = torch.bmm(span_start_logits.unsqueeze(-1), span_end_logits.unsqueeze(1))
        start_end_matrix = start_end_matrix.data.cpu().numpy()

        probs = []
        end_index = []
        for b in range(batch_size):
            max_span_log_prob = 0
            batch_prob = []
            batch_index = []
            for j in range(passage_length):
                range_prob = start_end_matrix[b, j, j:j+max_answer_span]
                index = np.argmax(range_prob)

                batch_prob.append(range_prob[index])
                batch_index.append(j + index)

                if range_prob[index] > max_span_log_prob:
                    max_span_log_prob = range_prob[index]
                    best_word_span[b, 0] = j
                    best_word_span[b, 1] = j + int(index)
            probs.append(batch_prob)
            end_index.append(batch_index)
        probs = np.array(probs)
        end_index = np.array(end_index)
        sorted_probs = np.argsort(probs)
        top_start = sorted_probs[:,-candidate_num:]
        # print(sorted_probs[0,-1])
        # print(best_word_span[0, 0])
        for b in range(batch_size):
            for j in range(candidate_num - 1):
                best_word_span[b, 2 * (j + 1)] = int(top_start[b, candidate_num - 2 - j])
                best_word_span[b, 2 * (j + 1) + 1] = int(end_index[b,top_start[b, candidate_num - 2 - j]])
    
        # span_start_logits = span_start_logits.data.cpu().numpy()
        # span_end_logits = span_end_logits.data.cpu().numpy()
                
        # for b in range(batch_size):  # pylint: disable=invalid-name
        #     for j in range(passage_length):
        #         val1 = span_start_logits[b, span_start_argmax[b]]
        #         if val1 < span_start_logits[b, j]:
        #             span_start_argmax[b] = j
        #             val1 = span_start_logits[b, j]

        #         val2 = span_end_logits[b, j]

        #         if val1 + val2 > max_span_log_prob[b]:
        #             best_word_span[b, 0] = span_start_argmax[b]
        #             best_word_span[b, 1] = j
        #             max_span_log_prob[b] = val1 + val2
        return best_word_span

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'BidirectionalAttentionFlow':
        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(vocab, embedder_params)
        summary_layer = Seq2SeqEncoder.from_params(params.pop("summery_layer"))
        phrase_layer = Seq2SeqEncoder.from_params(params.pop("phrase_layer"))
        residual_encoder = Seq2SeqEncoder.from_params(params.pop("residual_encoder"))
        span_start_encoder = Seq2SeqEncoder.from_params(params.pop("span_start_encoder"))
        span_end_encoder = Seq2SeqEncoder.from_params(params.pop("span_end_encoder"))
        initializer = InitializerApplicator.from_params(params.pop("initializer", []))
        dropout = params.pop('dropout', 0.2)

        # TODO: Remove the following when fully deprecated
        evaluation_json_file = params.pop('evaluation_json_file', None)
        if evaluation_json_file is not None:
            logger.warning("the 'evaluation_json_file' model parameter is deprecated, please remove")

        mask_lstms = params.pop('mask_lstms', True)
        params.assert_empty(cls.__name__)
        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   phrase_layer=phrase_layer,
                   summary_layer=summary_layer,
                   residual_encoder=residual_encoder,
                   span_start_encoder=span_start_encoder,
                   span_end_encoder=span_end_encoder,
                   initializer=initializer,
                   dropout=dropout,
                   mask_lstms=mask_lstms)
