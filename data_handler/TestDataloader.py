import sys
import traceback
import logging
import random
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from torch.utils.data import IterableDataset
from data_handler.streaming import StreamSamplerTest, get_files


def news_sample(news, ratio):
    if ratio > len(news):
        return news + [0] * (ratio - len(news))
    else:
        return random.sample(news, ratio)


class DataLoaderTest(IterableDataset):
    def __init__(self,
                 data_dirs,
                 filename_pat,
                 args,
                 world_size,
                 worker_rank,
                 cuda_device_idx,
                 news_index,
                 news_scoring,
                 enable_prefetch=True,
                 enable_shuffle=False,
                 enable_gpu=True):
        self.data_dirs = data_dirs
        self.filename_pat = filename_pat

        self.args = args
        self.worker_rank = worker_rank
        self.world_size = world_size
        self.cuda_device_idx = cuda_device_idx
        # data loader only cares about the config after tokenization.
        self.sampler = None

        self.enable_prefetch = enable_prefetch
        self.enable_shuffle = enable_shuffle
        self.enable_gpu = enable_gpu
        self.epoch = -1

        self.news_scoring = news_scoring
        self.news_index = news_index

        self.end = False

    def start_async(self):
        self.aval_count = 0
        self.stopped = False
        self.outputs = Queue(10)
        self.pool = ThreadPoolExecutor(1)
        self.pool.submit(self._produce)

    def __iter__(self):
        """Implement IterableDataset method to provide data iterator."""
        logging.info("DataLoader __iter__()")
        if self.enable_prefetch:
            self.join()
            self.start_async()
        else:
            self.start()
        return self


    def join(self):
        self.stopped = True
        if self.sampler:
            if self.enable_prefetch:
                while self.outputs.qsize() > 0:
                    self.outputs.get()
                    self.outputs.task_done()
                self.outputs.join()
                self.pool.shutdown(wait=True)
                logging.info("shut down pool.")
            self.sampler = None

    def start(self):
        self.epoch += 1
        self.sampler = StreamSamplerTest(
            data_dirs=self.data_dirs,
            filename_pat=self.filename_pat,
            batch_size=self.args.batch_size,
            worker_rank=self.worker_rank,
            world_size=self.world_size,
            enable_shuffle=self.enable_shuffle,
            shuffle_seed=self.epoch,  # epoch id as shuffle random seed
        )
        self.sampler.__iter__()


    # def __next__(self):
    #     if self.end and self.aval_count == 0:
    #         raise StopIteration

    #     if self.enable_prefetch:
    #         next_batch = self.outputs.get()
    #         self.outputs.task_done()
    #         self.aval_count -= 1
    #     else:
    #         next_batch = self._process(self.sampler.__next__())
    #     return next_batch
    def __next__(self):
        # Nếu đã hết dữ liệu và hàng đợi trống thì dừng
        if self.end and self.outputs.empty():
            raise StopIteration

        if self.enable_prefetch:
            while True:
                if self.end and self.outputs.empty():
                    raise StopIteration
                try:
                    # Wait 1s then check status again
                    next_batch = self.outputs.get(timeout=1) 
                    self.outputs.task_done()
                    self.aval_count -= 1
                    return next_batch
                except:
                    # Timeout (empty), check loop condition again
                    continue
        else:
            next_batch = self._process(self.sampler.__next__())
        return next_batch

    def trans_to_nindex(self, nids):
        # self.news_index đã là dict (được truyền từ news_info.news_index)
        target_dict = self.news_index
        return [target_dict[i] if i in target_dict else 0 for i in nids]

    def pad_to_fix_len(self, x, fix_length, padding_front=True, padding_value=0):
        if padding_front:
            pad_x = [padding_value] * (fix_length-len(x)) + x[-fix_length:]
            mask = [0] * (fix_length-len(x)) + [1] * min(fix_length, len(x))
        else:
            pad_x = x[-fix_length:] + [padding_value]*(fix_length-len(x))
            mask = [1] * min(fix_length, len(x)) + [0] * (fix_length - len(x))
        return pad_x, mask

    def _produce(self):
        # need to reset cuda device in produce thread.
        if self.enable_gpu:
            torch.cuda.set_device(self.cuda_device_idx)
        try:
            self.epoch += 1
            self.sampler = StreamSamplerTest(
                data_dirs=self.data_dirs,
                filename_pat=self.filename_pat,
                batch_size=self.args.batch_size,
                worker_rank=self.worker_rank,
                world_size=self.world_size,
                enable_shuffle=self.enable_shuffle,
                shuffle_seed=self.epoch,  # epoch id as shuffle random seed
            )
            # t0 = time.time()
            for batch in self.sampler:
                if self.stopped:
                    break
                context = self._process(batch)
                # Skip None results (empty batches)
                if context is not None:
                    self.outputs.put(context)
                    self.aval_count += 1
                # logging.info(f"_produce cost:{time.time()-t0}")
                # t0 = time.time()
            self.end = True
        except:
            traceback.print_exc(file=sys.stdout)
            self.pool.shutdown(wait=False)
            raise

    # def _process(self, batch):
    #     batch = [x.decode(encoding="utf-8").split("\t") for x in batch]

    #     user_feature_batch, log_mask_batch, news_feature_batch, news_bias_batch, label_batch = [], [], [], [], []

    #     for line in batch:
    #         click_docs = [i for i in line[2].split()]

    #         click_docs, log_mask  = self.pad_to_fix_len(self.trans_to_nindex(click_docs),
    #                                          self.args.user_log_length)
    #         user_feature = self.news_scoring[click_docs]

    #         sess_pos = [i.split(":")[0] for i in line[3].split()]
    #         sess_neg = [i for i in line[4].split()]
    #         sess_pos = self.trans_to_nindex(sess_pos)
    #         sess_neg = self.trans_to_nindex(sess_neg)

    #         sample_news = sess_pos + sess_neg

    #         labels = [1] * len(sess_pos) + [0] * len(sess_neg)

    #         news_feature = self.news_scoring[sample_news]
    #         user_feature_batch.append(user_feature)
    #         log_mask_batch.append(log_mask)
    #         news_feature_batch.append(news_feature)
    #         label_batch.append(np.array(labels))

    #     if self.enable_gpu:
    #         user_feature_batch = torch.FloatTensor(user_feature_batch).cuda()
    #         log_mask_batch = torch.FloatTensor(log_mask_batch).cuda()

    #     else:
    #         user_feature_batch = torch.FloatTensor(user_feature_batch)
    #         log_mask_batch = torch.FloatTensor(log_mask_batch)

    #     return user_feature_batch, log_mask_batch, news_feature_batch, label_batch
    def _process(self, batch):
        # Tách cột an toàn
        decoded_batch = []
        for x in batch:
            line = x.decode(encoding="utf-8").split("\t")
            if len(line) >= 5: # Đảm bảo đủ cột ImpressionID, UserID, History, Pos, Neg
                decoded_batch.append(line)

        user_feature_batch, log_mask_batch, news_feature_batch, label_batch = [], [], [], []

        for line in decoded_batch:
            try:
                click_docs_raw = line[2].split() # History
                
                # Chuyển đổi và padding history
                click_docs, log_mask = self.pad_to_fix_len(
                    self.trans_to_nindex(click_docs_raw),
                    self.args.user_log_length
                )
                
                # Lấy embedding của tin đã đọc từ news_scoring
                user_feature = self.news_scoring[click_docs]

                # Xử lý tin Click (Positive) và Non-click (Negative)
                # MIND format: line[3] thường là 'N123:1' hoặc 'N123'
                sess_pos_raw = [i.split(":")[0] for i in line[3].split()]
                sess_neg_raw = [i for i in line[4].split()]
                
                sess_pos = self.trans_to_nindex(sess_pos_raw)
                sess_neg = self.trans_to_nindex(sess_neg_raw)

                sample_news = sess_pos + sess_neg
                labels = [1] * len(sess_pos) + [0] * len(sess_neg)

                # Trích xuất news features
                news_feature = self.news_scoring[sample_news]
                
                user_feature_batch.append(user_feature)
                log_mask_batch.append(log_mask)
                news_feature_batch.append(news_feature)
                label_batch.append(np.array(labels))
            except Exception as e:
                continue # Bỏ qua dòng lỗi

        if not user_feature_batch:
            return None # Trả về None nếu batch rỗng

        # Chuyển sang Tensor
        if self.enable_gpu:
            user_feature_batch = torch.FloatTensor(np.array(user_feature_batch)).cuda()
            log_mask_batch = torch.FloatTensor(np.array(log_mask_batch)).cuda()
        else:
            user_feature_batch = torch.FloatTensor(np.array(user_feature_batch))
            log_mask_batch = torch.FloatTensor(np.array(log_mask_batch))

        return user_feature_batch, log_mask_batch, news_feature_batch, label_batch



class DataLoaderLeader(DataLoaderTest):
    def __init__(self,
                 data_dirs,
                 filename_pat,
                 args,
                 world_size,
                 worker_rank,
                 cuda_device_idx,
                 news_index,
                 news_scoring,
                 enable_prefetch=True,
                 enable_shuffle=False,
                 enable_gpu=True):
        self.args = args
        self.worker_rank = worker_rank
        self.world_size = world_size
        self.cuda_device_idx = cuda_device_idx
        # data loader only cares about the config after tokenization.
        self.sampler = None

        self.enable_prefetch = enable_prefetch
        self.enable_shuffle = enable_shuffle
        self.enable_gpu = enable_gpu
        self.epoch = -1

        self.news_scoring = news_scoring
        self.news_index = news_index

        self.end = False

        self.test_files = get_files(data_dirs, filename_pat)


    def generate_batch(self):
        user_feature_batch, log_mask_batch, news_feature_batch, news_bias_batch, label_batch, market_batch = [], [], [], [], [], []
        impids = []
        for file in self.test_files:
            print(f'predicting: {file}')
            for line in open(file, 'r'):
                impid, uid, history, impressions = line.strip().split('\t')
                click_docs = [i for i in history.split()]

                click_docs, log_mask  = self.pad_to_fix_len(self.trans_to_nindex(click_docs),
                                                 self.args.user_log_length)
                user_feature = self.news_scoring[click_docs]

                sess = [i for i in impressions.split()]
                sess_candidate = self.trans_to_nindex(sess)

                news_feature = self.news_scoring[sess_candidate]

                impids.append(impid)
                user_feature_batch.append(user_feature)
                log_mask_batch.append(log_mask)
                news_feature_batch.append(news_feature)

                if len(impids)==self.args.batch_size:
                    user_feature_batch = torch.FloatTensor(user_feature_batch).cuda()
                    log_mask_batch = torch.FloatTensor(log_mask_batch).cuda()
                    yield impids, user_feature_batch, log_mask_batch, news_feature_batch

                    impids, user_feature_batch, log_mask_batch, news_feature_batch = [], [], [], []

        if len(impids)>0:
            user_feature_batch = torch.FloatTensor(user_feature_batch).cuda()
            log_mask_batch = torch.FloatTensor(log_mask_batch).cuda()
            yield impids, user_feature_batch, log_mask_batch, news_feature_batch