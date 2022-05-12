from collections import Counter

import numpy as np
# import tensorflow as tf


def export_to_file(export_file_path, tokens, ner_tags, stc_idxs):
    print('x', stc_idxs)
    with open(export_file_path, "w", encoding='utf-8') as f:
        for i in range(0, len(tokens)):
            tags = ner_tags[i]
            toks = tokens[i]
            # idxs = stc_idxs[i]
            print('t',tags)
            if len(tokens) > 0:
                f.write(
                    # str(idxs)
                    # + "\t"
                    str(len(toks))
                    + "\t"
                    + "\t".join(toks)
                    + "\t"
                    + "\t".join(map(str, tags))
                    + "\n"
                )


# def map_record_to_training_data(record):
#     record = tf.strings.split(record, sep="\t")
#     length = tf.strings.to_number(record[0], out_type=tf.int32)
#     tokens = record[1: length + 1]
#     tags = record[length + 1:]
#     tags = tf.strings.to_number(tags, out_type=tf.int64)
#     tags += 1
#     #     print(tokens)
#     return tokens, tags


def unique_word_count(tokenize_word_list):
    # Nối tất cả các từ trong tất cả các câu thành 1 list
    all_tokens = tokenize_word_list
    # Viết thường các chữ
    all_tokens_array = np.array(list(map(str.lower, all_tokens)))
    # Đếm số từ unique <- dùng hàm Counter để đếm giá trị unique có bao nhiêu cái trùng <- lấy len của counter sẽ ra
    unique_word_count_counter = Counter(all_tokens_array)
    print('Số lượng từ unique: ', len(unique_word_count_counter))

    return unique_word_count_counter


# def lookup_layer(counter, vocab_size):
#     """
#
#     :param vocab_size: số lượng từ vocab được chọn
#     :param counter: counter return từ hàm unique_word_count để lấy most_common word
#     :param num_tags: num tag types
#     :param tokenize_word_list: word of all sentences list
#     """
#
#     # We only take (vocab_size - 2) most commons words from the training data since
#     # the `StringLookup` class uses 2 additional tokens - one denoting an unknown
#     # token and another one denoting a masking token
#     vocabulary = [token for token, count in counter.most_common(
#         vocab_size - 2)]  # counter có hàm most_common để lấy n giá trị trùng lặp nhiều nhất
#
#     # The StringLook class will convert tokens to token IDs
#     StringLookup_layer = tf.keras.layers.StringLookup(
#         vocabulary=vocabulary
#     )
#     return StringLookup_layer
#
#
# def lowercase_and_convert_to_ids(tokens, vocab_size):
#     tokens = tf.strings.lower(tokens)
#     unique_word_count_counter = unique_word_count()
#     StringLookup_layer = lookup_layer(unique_word_count_counter, vocab_size)
#     return StringLookup_layer(tokens)
