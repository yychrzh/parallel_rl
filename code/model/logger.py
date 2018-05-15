import tensorflow as tf


class Logger(object):
    def __init__(self, session, log_path, name_list, type_list):
        self.session = session
        self.log_path = log_path
        self.name_list = name_list
        self.type_list = type_list
        self.variable_num = len(self.name_list)
        self.train_writer = tf.summary.FileWriter(self.log_path, self.session.graph)

    def create_scalar_log_method(self):
        log_variable = []
        for i in range(self.variable_num):
            var = tf.placeholder(dtype=self.type_list[i])
            log_variable.append(var)
            tf.summary.scalar(self.name_list[i], var)
        # summaries merged
        merged = tf.summary.merge_all()

        def save_summary(summary_data, step):
            feed_dict = {}
            for i in range(len(summary_data)):
                dict_elment = {log_variable[i]: summary_data[i]}
                feed_dict.update(dict_elment)
            summary = self.session.run(merged, feed_dict=feed_dict)
            self.train_writer.add_summary(summary, step)
        return save_summary