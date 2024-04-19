from bert_score import score
from rouge import Rouge


def rouge_score(ref, pred):
    rouge = Rouge(metrics=["rouge-1", "rouge-2", "rouge-3", "rouge-l"])
    rs = rouge.get_scores(pred, ref)
    rouge1 = rs[0]["rouge-1"]["f"] * 100
    rouge2 = rs[0]["rouge-2"]["f"] * 100
    rouge3 = rs[0]["rouge-3"]["f"] * 100
    rougel = rs[0]["rouge-l"]["f"] * 100
    return rouge1, rouge2, rouge3, rougel


def bs_score(ref, pred):
    _, _, F1 = score([pred], [ref], lang="en", verbose=True)
    bs = F1.mean()
    return bs


class BatchEvaluation:
    def __init__(
        self,
        total_r1=0,
        total_r2=0,
        total_r3=0,
        total_rl=0,
        total_bs=0,
        call_time_rs=0,
        call_time_bs=0,
    ):
        self.ref = ""
        self.pred = ""

        self.total_r1 = total_r1
        self.total_r2 = total_r2
        self.total_r3 = total_r3
        self.total_rl = total_rl
        self.total_bs = total_bs
        self.call_time_rs = call_time_rs
        self.call_time_bs = call_time_bs

    def set_text(self, ref, pred):
        self.ref = ref
        self.pred = pred
        return self

    def get_rouge_score(self):
        r1, r2, r3, rl = rouge_score(self.ref, self.pred)
        self.total_r1 += r1
        self.total_r2 += r2
        self.total_r3 += r3
        self.total_rl += rl
        self.call_time_rs += 1

    def get_bs_score(self):
        bs = bs_score(self.ref, self.pred)
        self.total_bs += bs
        self.call_time_bs += 1
