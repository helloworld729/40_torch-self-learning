""" Statistics calculation utility """
from __future__ import division

import sys
import time

from others.logging import logger


class Statistics(object):
    """
    Accumulator for loss statistics.
    Currently calculates:

    * accuracy
    * perplexity
    * elapsed time
    """

    def __init__(self, loss=0, n_docs=0, n_correct=0):
        self.loss = loss
        self.n_docs = n_docs
        self.start_time = time.time()

    def update(self, stat, update_n_src_words=False):
        """
        Update statistics by suming values with another `Statistics` object

        Args:
            stat: another statistic object
            update_n_src_words(bool): whether to update (sum) `n_src_words`
                or not

        """
        self.loss += stat.loss

        self.n_docs += stat.n_docs

    def xent(self):
        """ compute cross entropy """
        if(self.n_docs==0):
            return 0
        return self.loss/self.n_docs


    def elapsed_time(self):
        """ compute elapsed time """
        return time.time() - self.start_time

    def output(self, step, num_steps, learning_rate, start):
        """Write out statistics to stdout.

        Args:
           step (int): current step
           n_batch (int): total batches
           start (int): start time of step.
        """
        t = self.elapsed_time()
        step_fmt = "%2d" % step
        if num_steps > 0:
            step_fmt = "%s/%5d" % (step_fmt, num_steps)
        logger.info(
            ("Step %s; xent: %4.2f; " +
             "lr: %7.7f; %3.0f docs/s; %6.0f sec")
            % (step_fmt,
               self.xent(),
               learning_rate,
               self.n_docs / (t + 1e-5),
               time.time() - start))
        sys.stdout.flush()

    def log_tensorboard(self, prefix, writer, learning_rate, step):
        """ display statistics to tensorboard """
        t = self.elapsed_time()
        writer.add_scalar(prefix + "/xent", self.xent(), step)
        writer.add_scalar(prefix + "/lr", learning_rate, step)
