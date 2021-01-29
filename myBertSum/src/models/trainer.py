import os
from tqdm import tqdm
import numpy as np
import torch
from tensorboardX import SummaryWriter

from models.data_loader import load_dataset
from models import data_loader, model_builder
from models.reporter import ReportMgr
from models.stats import Statistics
from others.logging import logger


def _tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    return n_params


def build_trainer(args, model=None, optim=None):

    tensorboard_log_dir = args.model_path
    writer = SummaryWriter(tensorboard_log_dir)
    report_manager = ReportMgr(args.report_every,
                    start_time=-1, tensorboard_writer=writer)

    trainer = Trainer(args, model, optim, report_manager)
    if model:
        n_params = _tally_parameters(model)
        logger.info('* number of parameters: %d' % n_params)

    return trainer


class Trainer(object):
    def __init__(self,  args, model, optim, report_manager=None):
        self.args = args
        self.save_checkpoint_steps = args.save_checkpoint_steps
        self.model = model
        self.optim = optim
        self.report_manager = report_manager
        self.loss = torch.nn.BCELoss(reduction='none')

        # Set model in training mode.
        if (model): self.model.train()

    def train_iter_fct(self):  # 甘泉宫
        device = "cpu" if not torch.cuda.is_available() else "cuda"
        # 源数据生成器
        dataSet = load_dataset(self.args, 'train', shuffle=True)

        return data_loader.Dataloader(self.args, dataSet, device,
                                      shuffle=True, is_test=False)

    def train(self, train_steps, valid_iter_fct=None, valid_steps=-1):
        # train_steps:全局步数
        logger.info('Start training...')

        step = self.optim._step + 1
        normalization = 0

        data_loader = self.train_iter_fct()

        total_stats = Statistics()
        report_stats = Statistics()
        self._start_report_manager(start_time=total_stats.start_time)

        while step <= train_steps:

            # 下面的for循环表示一个epoch
            # for _, batch in enumerate(data_loader):
            print("step/totalSteps:", step, "/", train_steps)
            bar = tqdm(data_loader, leave=True)
            for batch in bar:

                normalization += batch.batch_size

                loss = self._backPropagation(
                    batch, normalization, total_stats,
                    report_stats, step)
                bar.set_description("loss: %10.8s" % loss)

                report_stats = self._maybe_report_training(
                    step, train_steps,
                    self.optim.learning_rate,
                    report_stats)

                normalization = 0
                if (step % self.save_checkpoint_steps == 0):
                    self._save(step)

                step += 1
                if step > train_steps:
                    break
            data_loader = self.train_iter_fct()

        return total_stats

    def validate(self, valid_iter, step=0):
        self.model.eval()
        stats = Statistics()

        with torch.no_grad():
            for batch in valid_iter:

                src = batch.src
                labels = batch.labels
                segs = batch.segs
                clss = batch.clss
                mask = batch.mask
                mask_cls = batch.mask_cls

                sent_scores, mask = self.model(src, segs, clss, mask, mask_cls)

                loss = self.loss(sent_scores, labels.float())
                loss = (loss * mask.float()).sum()
                batch_stats = Statistics(float(loss.cpu().data.numpy()), len(labels))
                stats.update(batch_stats)
            self._report_step(0, step, valid_stats=stats)
            return stats

    def test(self, test_iter, step, cal_lead=False, cal_oracle=False):
        """
        测试函数
        :param test_iter: 测试数据生成器
        :param step: 加载多少步的checkPoint
        :param cal_lead: 前三句摘要
        :param cal_oracle: Oracle摘要
        :return: None
        """
        # Set model in validating mode.
        def _get_ngrams(n, text):
            ngram_set = set()
            text_length = len(text)
            max_index_ngram_start = text_length - n
            for i in range(max_index_ngram_start + 1):
                ngram_set.add(tuple(text[i:i + n]))
            return ngram_set

        def _block_tri(c, p):
            tri_c = _get_ngrams(3, c.split())
            for s in p:
                tri_s = _get_ngrams(3, s.split())
                if len(tri_c.intersection(tri_s))>0:
                    return True
            return False

        if not cal_lead and not cal_oracle:
            self.model.eval()
        stats = Statistics()

        # 候选摘要地址 result文件夹
        can_path = '%s_step%d.candidate'%(self.args.result_path, step)
        # 最佳摘要地址 result文件夹
        gold_path = '%s_step%d.gold' % (self.args.result_path, step)

        with open(can_path, 'w') as save_pred, open(gold_path, 'w') as save_gold:
            with torch.no_grad():
                for batch in test_iter:
                    src = batch.src
                    labels = batch.labels
                    segs = batch.segs
                    clss = batch.clss
                    mask = batch.mask
                    mask_cls = batch.mask_cls

                    gold = []  # batch 最佳容器
                    pred = []  # batch 候选容器

                    if (cal_lead):
                        selected_ids = [list(range(batch.clss.size(1)))] * \
                                       batch.batch_size
                    elif (cal_oracle):
                        selected_ids = [[j for j in range(batch.clss.size(1))
                                         if labels[i][j] == 1] for i in
                                        range(batch.batch_size)]
                    else:
                        # 获取句子打分
                        sent_scores, mask = self.model(src, segs, clss, mask, mask_cls)

                        loss = self.loss(sent_scores, labels.float())
                        loss = (loss * mask.float()).sum()
                        batch_stats = Statistics(float(loss.cpu().data.numpy()), len(labels))
                        stats.update(batch_stats)

                        sent_scores = sent_scores + mask.float()
                        sent_scores = sent_scores.cpu().data.numpy()

                        selected_ids = np.argsort(-sent_scores, 1)  # 倒排
                    # selected_ids = np.sort(selected_ids,1)

                    for i, idx in enumerate(selected_ids):
                        _pred = []
                        if(len(batch.src_str[i])==0):
                            continue
                        for j in selected_ids[i][:len(batch.src_str[i])]:
                            if(j>=len( batch.src_str[i])):
                                continue
                            # 生成 候选句
                            candidate = batch.src_str[i][j].strip()
                            if self.args.block_trigram:
                                if not _block_tri(candidate,_pred):
                                    _pred.append(candidate)
                            else:
                                _pred.append(candidate)

                            # 生成3句候选句
                            if not cal_oracle and not self.args.recall_eval \
                                    and len(_pred) == 3:
                                break

                        _pred = '<q>'.join(_pred)  # 字符串

                        if self.args.recall_eval:
                            _pred = ' '.join(_pred.split()[:len(batch.tgt_str[i].split())])

                        pred.append(_pred)
                        gold.append(batch.tgt_str[i])

                    for i in range(len(gold)):
                        save_gold.write(gold[i].strip()+'\n')
                    for i in range(len(pred)):
                        save_pred.write(pred[i].strip()+'\n')

        self._report_step(0, step, valid_stats=stats)

        return stats

    def _backPropagation(self, batch, normalization, total_stats,
                               report_stats, step):
        src = batch.src
        labels = batch.labels
        segs = batch.segs
        clss = batch.clss
        mask = batch.mask
        mask_cls = batch.mask_cls

        sent_scores, mask = self.model(src, segs, clss, mask, mask_cls)

        loss = self.loss(sent_scores, labels.float())
        loss = (loss*mask.float()).sum()
        (loss/loss.numel()).backward()
        # loss.div(float(normalization)).backward()

        batch_stats = Statistics(float(loss.cpu().data.numpy()), normalization)

        total_stats.update(batch_stats)
        report_stats.update(batch_stats)

        self.optim.learning_rate = pow(step, -0.5) * self.args.lr
        self.optim.step()
        return loss.item()

    def _save(self, step):
        real_model = self.model
        # real_generator = (self.generator.module
        #                   if isinstance(self.generator, torch.nn.DataParallel)
        #                   else self.generator)

        model_state_dict = real_model.state_dict()
        # generator_state_dict = real_generator.state_dict()
        checkpoint = {
            'model': model_state_dict,
            # 'generator': generator_state_dict,
            'opt': self.args,
            'optim': self.optim,
        }
        checkpoint_path = os.path.join(self.args.model_path, 'model_step_%d.pt' % step)
        logger.info("Saving checkpoint %s" % checkpoint_path)
        # checkpoint_path = '%s_step_%d.pt' % (FLAGS.model_path, step)
        if (not os.path.exists(checkpoint_path)):
            torch.save(checkpoint, checkpoint_path)
            return checkpoint, checkpoint_path

    def _start_report_manager(self, start_time=None):
        """
        Simple function to start report manager (if any)
        """
        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time


    def _maybe_report_training(self, step, num_steps, learning_rate,
                               report_stats):
        """
        Simple function to report training stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_training` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_training(
                step, num_steps, learning_rate, report_stats,
                multigpu=False)

    def _report_step(self, learning_rate, step, train_stats=None,
                     valid_stats=None):
        """
        Simple function to report stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_step` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_step(
                learning_rate, step, train_stats=train_stats,
                valid_stats=valid_stats)

