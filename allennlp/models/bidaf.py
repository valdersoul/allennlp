import logging
from typing import Any, Dict, List

import torch
from torch.autograd import Variable
from torch.nn.functional import nll_loss, log_softmax, cross_entropy, softmax

from allennlp.common import Params, squad_eval
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Highway, MatrixAttention
from allennlp.modules import Seq2SeqEncoder, SimilarityFunction, TimeDistributed, TextFieldEmbedder
from allennlp.nn import InitializerApplicator, util
from allennlp.training.metrics import Average, BooleanAccuracy, CategoricalAccuracy
from allennlp.models.interweighted import interWeighted, finalModel

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("bidaf")
class BidirectionalAttentionFlow(Model):
    """
    This class implements Minjoon Seo's `Bidirectional Attention Flow model
    <https://www.semanticscholar.org/paper/Bidirectional-Attention-Flow-for-Machine-Seo-Kembhavi/7586b7cca1deba124af80609327395e613a20e9d>`_
    for answering reading comprehension questions (ICLR 2017).

    The basic layout is pretty simple: encode words as a combination of word embeddings and a
    character-level encoder, pass the word representations through a bi-LSTM/GRU, use a matrix of
    attentions to put question information into the passage word representations (this is the only
    part that is at all non-standard), pass this through another few layers of bi-LSTMs/GRUs, and
    do a softmax over span start and span end.

    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the ``question`` and ``passage`` ``TextFields`` we get as input to the model.
    num_highway_layers : ``int``
        The number of highway layers to use in between embedding the input and passing it through
        the phrase layer.
    phrase_layer : ``Seq2SeqEncoder``
        The encoder (with its own internal stacking) that we will use in between embedding tokens
        and doing the bidirectional attention.
    attention_similarity_function : ``SimilarityFunction``
        The similarity function that we will use when comparing encoded passage and question
        representations.
    modeling_layer : ``Seq2SeqEncoder``
        The encoder (with its own internal stacking) that we will use in between the bidirectional
        attention and predicting span start and end.
    span_end_encoder : ``Seq2SeqEncoder``
        The encoder that we will use to incorporate span start predictions into the passage state
        before predicting span end.
    initializer : ``InitializerApplicator``
        We will use this to initialize the parameters in the model, calling ``initializer(self)``.
    dropout : ``float``, optional (default=0.2)
        If greater than 0, we will apply dropout with this probability after all encoders (pytorch
        LSTMs do not apply dropout to their last layer).
    mask_lstms : ``bool``, optional (default=True)
        If ``False``, we will skip passing the mask to the LSTM layers.  This gives a ~2x speedup,
        with only a slight performance decrease, if any.  We haven't experimented much with this
        yet, but have confirmed that we still get very similar performance with much faster
        training times.  We still use the mask for all softmaxes, but avoid the shuffling that's
        required when using masking with pytorch LSTMs.
    evaluation_json_file : ``str``, optional
        If given, we will load this JSON into memory and use it to compute official metrics
        against.  We need this separately from the validation dataset, because the official metrics
        use all of the annotations, while our dataset reader picks the most frequent one.
    """
    def __init__(self, vocab: Vocabulary,
                 #text_field_embedder: TextFieldEmbedder,
                 num_highway_layers: int,
                 phrase_layer: Seq2SeqEncoder,
                 attention_similarity_function: SimilarityFunction,
                 modeling_layer: Seq2SeqEncoder,
                 span_end_encoder: Seq2SeqEncoder,
                 initializer: InitializerApplicator,
                 sentence_embedder: TextFieldEmbedder = None,
                 dropout: float = 0.2,
                 mask_lstms: bool = True) -> None:
        super(BidirectionalAttentionFlow, self).__init__(vocab)

        #self._text_field_embedder = text_field_embedder
        self._sentence_embedder = sentence_embedder
        # self._highway_layer = TimeDistributed(Highway(text_field_embedder.get_output_dim(),
        #                                               num_highway_layers))

        self._inter_attention = interWeighted(200)
        self._final_model = finalModel(800, 3)

        self._phrase_layer = phrase_layer
        self._matrix_attention = MatrixAttention(attention_similarity_function)
        #self._4d_matrix_attention = TimeDistributed(MatrixAttention(attention_similarity_function))
        #self._modeling_layer = modeling_layer
        #self._span_end_encoder = span_end_encoder

        encoding_dim = phrase_layer.get_output_dim()
        modeling_dim = modeling_layer.get_output_dim()
        span_start_input_dim = encoding_dim * 4 + modeling_dim
        self._span_start_predictor = TimeDistributed(torch.nn.Linear(span_start_input_dim, 1))

        span_end_encoding_dim = span_end_encoder.get_output_dim()
        span_end_input_dim = encoding_dim * 4 + span_end_encoding_dim
        #self._span_end_predictor = TimeDistributed(torch.nn.Linear(span_end_input_dim, 1))
        initializer(self)

        # Bidaf has lots of layer dimensions which need to match up - these
        # aren't necessarily obvious from the configuration files, so we check
        # here.
        # if modeling_layer.get_input_dim() != 4 * encoding_dim:
        #     raise ConfigurationError("The input dimension to the modeling_layer must be "
        #                              "equal to 4 times the encoding dimension of the phrase_layer. "
        #                              "Found {} and 4 * {} respectively.".format(modeling_layer.get_input_dim(),
        #                                                                         encoding_dim))
        # if text_field_embedder.get_output_dim() != phrase_layer.get_input_dim():
        #     raise ConfigurationError("The output dimension of the text_field_embedder (embedding_dim + "
        #                              "char_cnn) must match the input dimension of the phrase_encoder. "
        #                              "Found {} and {}, respectively.".format(text_field_embedder.get_output_dim(),
        #                                                                      phrase_layer.get_input_dim()))

        # if span_end_encoder.get_input_dim() != encoding_dim * 4 + modeling_dim * 3:
        #     raise ConfigurationError("The input dimension of the span_end_encoder should be equal to "
        #                              "4 * phrase_layer.output_dim + 3 * modeling_layer.output_dim. "
        #                              "Found {} and (4 * {} + 3 * {}) "
        #                              "respectively.".format(span_end_encoder.get_input_dim(),
        #                                                     encoding_dim,
        #                                                     modeling_dim))

        self._span_start_accuracy = CategoricalAccuracy()
        self._span_end_accuracy = CategoricalAccuracy()
        self._span_accuracy = BooleanAccuracy()
        self._official_em = Average()
        self._official_f1 = Average()
        self._sentence_correct = Average()
        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x
        self._mask_lstms = mask_lstms

    def forward(self,  # type: ignore
                question: Dict[str, torch.LongTensor],
                passage: Dict[str, torch.LongTensor],
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
        # sentence embedding:
        embedded_passage_sentences = self._sentence_embedder(sentences)
        embedded_question_sentence = self._sentence_embedder(question_sentence)
        sentence_size = embedded_passage_sentences.size()
        question_size = embedded_question_sentence.size()
        embedded_question_sentence = embedded_question_sentence.repeat(1, sentence_size[1], 1, 1)
        embedded_question_sentence = embedded_question_sentence.view(-1, question_size[-2], question_size[-1])
        embedded_passage_sentences = embedded_passage_sentences.view(-1, sentence_size[-2], sentence_size[-1])

        batch_size = sentence_size[0]

        question_mask = util.get_text_field_mask(question_sentence).float().repeat(1, sentence_size[1], 1).view(embedded_passage_sentences.size()[0], -1)
        passage_mask = util.get_text_field_mask(sentences).float().view(embedded_passage_sentences.size()[0], -1)
 

        question_lstm_mask = question_mask if self._mask_lstms else None
        passage_lstm_mask = passage_mask if self._mask_lstms else None
        encoded_question = self._dropout(self._phrase_layer(embedded_question_sentence, question_lstm_mask))
        encoded_passage = self._dropout(self._phrase_layer(embedded_passage_sentences, passage_lstm_mask)) 

        sentence_similarity = self._matrix_attention(embedded_passage_sentences, embedded_question_sentence)
        weighted_passage = util.weighted_sum(encoded_question, util.last_dim_softmax(sentence_similarity, question_mask))
        question_similarity = sentence_similarity.permute(0,2,1).contiguous()
        weighted_question = util.weighted_sum(encoded_passage, util.last_dim_softmax(question_similarity, passage_mask))
        
        s_r = torch.abs(weighted_passage - encoded_passage)
        q_r = torch.abs(weighted_question - encoded_question)
        sentence_embedding_weight = util.masked_softmax(self._inter_attention(encoded_passage, encoded_question), passage_mask)
        question_embedding_weight = util.masked_softmax(self._inter_attention(encoded_question, encoded_passage), question_mask)

        swh = util.weighted_sum(encoded_passage, sentence_embedding_weight)
        twh = util.weighted_sum(encoded_question, question_embedding_weight)
        h_cross = swh * twh
        h_plus = torch.abs(swh - twh)
        swr = util.weighted_sum(s_r, sentence_embedding_weight)
        twr = util.weighted_sum(q_r, question_embedding_weight)

        feature = torch.cat((h_cross, h_plus, swr, twr), -1)
        logits = self._final_model(feature)

        correct_lable = correct_label.view(-1)
        weights = Variable(torch.zeros(3)).cuda()
        diff = torch.sum((correct_lable == 2).long()).float()
        weights[1] = diff / (torch.sum((correct_lable != 0).long()).float())
        weights[2] = 1- weights[1]

        # print(correct_lable)
        # print(logits)
        # print(weights)

        sentence_loss = cross_entropy(input=logits, target=correct_lable, weight=weights, ignore_index=0, size_average=True)

        true_score = softmax(logits)[:,2].contiguous()
        true_score = true_score.view(batch_size, -1)

        _, index = torch.max(true_score, -1)
        if index.size()[0] < batch_size:
            print(index)
            print(logits)
            print("index")
            print(correct_sentence)
        if correct_sentence.size()[0] < batch_size:
            print(correct_sentence)
            print("correct")
        index = index.data.cpu().numpy()
        
        for i in range(batch_size):
            self._sentence_correct(1 if index[i] in correct_sentence[i].data.cpu().numpy() else 0)
        # print(weighted_sentence.size())
        # print(weighted_question.size())
        # embedded_passage_sentences = embedded_passage_sentences.view(-1, embedded_passage_sentences.size(2), embedded_passage_sentences.size(3))
        # test_mask = util.get_text_field_mask(sentences).float()
        # print(embedded_passage_sentences.size())
        # print(test_mask.size())
        # test_mask = test_mask.view(-1, test_mask.size(2))
        # encoded_question = self._dropout(self._phrase_layer(embedded_passage_sentences, test_mask))
        # print(encoded_question.size())
        
        # embedded_question_sentence = self._sentence_embedder(question_sentence)

        # question_mask_s = util.get_text_field_mask(question_sentence).float()
        # passage_mask_s = util.get_text_field_mask(sentences).float()
        # test = TimeDistributed(self._phrase_layer(embedded_passage_sentences, passage_mask_s))
        # print(test.size())

        # sentence embedding end
        # embedded_question = self._highway_layer(self._text_field_embedder(question))
        # embedded_passage = self._highway_layer(self._text_field_embedder(passage))
        # batch_size = embedded_question.size(0)
        # passage_length = embedded_passage.size(1)
        # question_mask = util.get_text_field_mask(question).float()
        # passage_mask = util.get_text_field_mask(passage).float()
        # question_lstm_mask = question_mask if self._mask_lstms else None
        # passage_lstm_mask = passage_mask if self._mask_lstms else None

        # encoded_question = self._dropout(self._phrase_layer(embedded_question, question_lstm_mask))
        # encoded_passage = self._dropout(self._phrase_layer(embedded_passage, passage_lstm_mask))
        # encoding_dim = encoded_question.size(-1)

        # # Shape: (batch_size, passage_length, question_length)
        # passage_question_similarity = self._matrix_attention(encoded_passage, encoded_question)
        # # Shape: (batch_size, passage_length, question_length)
        # passage_question_attention = util.last_dim_softmax(passage_question_similarity, question_mask)
        # # Shape: (batch_size, passage_length, encoding_dim)
        # passage_question_vectors = util.weighted_sum(encoded_question, passage_question_attention)

        # # We replace masked values with something really negative here, so they don't affect the
        # # max below.
        # masked_similarity = util.replace_masked_values(passage_question_similarity,
        #                                                question_mask.unsqueeze(1),
        #                                                -1e7)
        # # Shape: (batch_size, passage_length)
        # question_passage_similarity = masked_similarity.max(dim=-1)[0].squeeze(-1)
        # # Shape: (batch_size, passage_length)
        # question_passage_attention = util.masked_softmax(question_passage_similarity, passage_mask)
        # # Shape: (batch_size, encoding_dim)
        # question_passage_vector = util.weighted_sum(encoded_passage, question_passage_attention)
        # # Shape: (batch_size, passage_length, encoding_dim)
        # tiled_question_passage_vector = question_passage_vector.unsqueeze(1).expand(batch_size,
        #                                                                             passage_length,
        #                                                                             encoding_dim)

        # # Shape: (batch_size, passage_length, encoding_dim * 4)
        # final_merged_passage = torch.cat([encoded_passage,
        #                                   passage_question_vectors,
        #                                   encoded_passage * passage_question_vectors,
        #                                   encoded_passage * tiled_question_passage_vector],
        #                                  dim=-1)

        # modeled_passage = self._dropout(self._modeling_layer(final_merged_passage, passage_lstm_mask))
        # modeling_dim = modeled_passage.size(-1)

        # # Shape: (batch_size, passage_length, encoding_dim * 4 + modeling_dim))
        # span_start_input = self._dropout(torch.cat([final_merged_passage, modeled_passage], dim=-1))
        # # Shape: (batch_size, passage_length)
        # span_start_logits = self._span_start_predictor(span_start_input).squeeze(-1)
        # # Shape: (batch_size, passage_length)
        # span_start_probs = util.masked_softmax(span_start_logits, passage_mask)

        # # Shape: (batch_size, modeling_dim)
        # span_start_representation = util.weighted_sum(modeled_passage, span_start_probs)
        # # Shape: (batch_size, passage_length, modeling_dim)
        # tiled_start_representation = span_start_representation.unsqueeze(1).expand(batch_size,
        #                                                                            passage_length,
        #                                                                            modeling_dim)

        # # Shape: (batch_size, passage_length, encoding_dim * 4 + modeling_dim * 3)
        # span_end_representation = torch.cat([final_merged_passage,
        #                                      modeled_passage,
        #                                      tiled_start_representation,
        #                                      modeled_passage * tiled_start_representation],
        #                                     dim=-1)
        # # Shape: (batch_size, passage_length, encoding_dim)
        # encoded_span_end = self._dropout(self._span_end_encoder(span_end_representation,
        #                                                         passage_lstm_mask))
        # # Shape: (batch_size, passage_length, encoding_dim * 4 + span_end_encoding_dim)
        # span_end_input = self._dropout(torch.cat([final_merged_passage, encoded_span_end], dim=-1))
        # span_end_logits = self._span_end_predictor(span_end_input).squeeze(-1)
        # span_end_probs = util.masked_softmax(span_end_logits, passage_mask)
        # span_start_logits = util.replace_masked_values(span_start_logits, passage_mask, -1e7)
        # span_end_logits = util.replace_masked_values(span_end_logits, passage_mask, -1e7)
        # best_span = self._get_best_span(span_start_logits, span_end_logits)

        # output_dict = {"span_start_logits": span_start_logits,
        #                "span_start_probs": span_start_probs,
        #                "span_end_logits": span_end_logits,
        #                "span_end_probs": span_end_probs,
        #                "best_span": best_span}
        # if span_start is not None:
        #     loss = nll_loss(util.masked_log_softmax(span_start_logits, passage_mask), span_start.squeeze(-1))
        #     self._span_start_accuracy(span_start_logits, span_start.squeeze(-1))
        #     loss += nll_loss(util.masked_log_softmax(span_end_logits, passage_mask), span_end.squeeze(-1))
        #     self._span_end_accuracy(span_end_logits, span_end.squeeze(-1))
        #     self._span_accuracy(best_span, torch.stack([span_start, span_end], -1))
        output_dict = {}
        output_dict["loss"] = sentence_loss
        # if metadata is not None:
        #     output_dict['best_span_str'] = []
        #     for i in range(batch_size):
        #         passage_str = metadata[i]['original_passage']

        #         import nltk
        #         sentences = nltk.sent_tokenize(passage_str)

        #         offsets = metadata[i]['token_offsets']
        #         predicted_span = tuple(best_span[i].data.cpu().numpy())
        #         start_offset = offsets[predicted_span[0]][0]
        #         end_offset = offsets[predicted_span[1]][1]
        #         best_span_string = passage_str[start_offset:end_offset]

        #         index = []
        #         for j, sentence in enumerate(sentences):
        #             if best_span_string in sentence:
        #                 index.append(j+1)
        #         self._sentence_correct(1 if any(a in index for a in correct_sentence[i].data.cpu().numpy()) else 0)

        #         output_dict['best_span_str'].append(best_span_string)
        #         answer_texts = metadata[i].get('answer_texts', [])

        #         exact_match = f1_score = 0
        #         if answer_texts:
        #             exact_match = squad_eval.metric_max_over_ground_truths(
        #                     squad_eval.exact_match_score,
        #                     best_span_string,
        #                     answer_texts)
        #             f1_score = squad_eval.metric_max_over_ground_truths(
        #                     squad_eval.f1_score,
        #                     best_span_string,
        #                     answer_texts)
        #         self._official_em(100 * exact_match)
        #         self._official_f1(100 * f1_score)
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
                # 'start_acc': self._span_start_accuracy.get_metric(reset),
                # 'end_acc': self._span_end_accuracy.get_metric(reset),
                # 'span_acc': self._span_accuracy.get_metric(reset),
                # 'em': self._official_em.get_metric(reset),
                # 'f1': self._official_f1.get_metric(reset),
                'sentence': self._sentence_correct.get_metric(reset)
                }

    @staticmethod
    def _get_best_span(span_start_logits: Variable, span_end_logits: Variable) -> Variable:
        if span_start_logits.dim() != 2 or span_end_logits.dim() != 2:
            raise ValueError("Input shapes must be (batch_size, passage_length)")
        batch_size, passage_length = span_start_logits.size()
        max_span_log_prob = [-1e20] * batch_size
        span_start_argmax = [0] * batch_size
        best_word_span = Variable(span_start_logits.data.new()
                                  .resize_(batch_size, 2).fill_(0)).long()

        span_start_logits = span_start_logits.data.cpu().numpy()
        span_end_logits = span_end_logits.data.cpu().numpy()

        for b in range(batch_size):  # pylint: disable=invalid-name
            for j in range(passage_length):
                val1 = span_start_logits[b, span_start_argmax[b]]
                if val1 < span_start_logits[b, j]:
                    span_start_argmax[b] = j
                    val1 = span_start_logits[b, j]

                val2 = span_end_logits[b, j]

                if val1 + val2 > max_span_log_prob[b]:
                    best_word_span[b, 0] = span_start_argmax[b]
                    best_word_span[b, 1] = j
                    max_span_log_prob[b] = val1 + val2
        return best_word_span

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'BidirectionalAttentionFlow':
        #embedder_params = params.pop("text_field_embedder")
        sentence_embedder_params = params.pop("sentence_field_embedder")
        #text_field_embedder = TextFieldEmbedder.from_params(vocab, embedder_params)
        sentence_embedder = TextFieldEmbedder.from_params(vocab, sentence_embedder_params)
        num_highway_layers = params.pop("num_highway_layers")
        phrase_layer = Seq2SeqEncoder.from_params(params.pop("phrase_layer"))
        similarity_function = SimilarityFunction.from_params(params.pop("similarity_function"))
        modeling_layer = Seq2SeqEncoder.from_params(params.pop("modeling_layer"))
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
                   #text_field_embedder=text_field_embedder,
                   sentence_embedder=sentence_embedder,
                   num_highway_layers=num_highway_layers,
                   phrase_layer=phrase_layer,
                   attention_similarity_function=similarity_function,
                   modeling_layer=modeling_layer,
                   span_end_encoder=span_end_encoder,
                   initializer=initializer,
                   dropout=dropout,
                   mask_lstms=mask_lstms)
