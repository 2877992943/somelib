import math




##   症状结构化 规则预测    不支持batch
start_end_str2idx={'E临床表现': 4, 'S临床表现': 5, 'E否定': 8, 'S否定': 9, 'E时间描述': 10, 'S时间描述': 11, 'E身体部位': 12, 'S身体部位': 13, 'E前置条件': 16, 'S前置条件': 17, 'E方位': 20, 'S方位': 21, 'E程度': 34, 'S程度': 35, 'E频率和次数': 41, 'S频率和次数': 42, 'E限定值': 49, 'S限定值': 50, 'E趋势': 51, 'S趋势': 52, 'E范围大小': 90, 'S范围大小': 91, 'E颜色': 129, 'S颜色': 130, 'E形状': 401, 'S形状': 402, 'E气味': 591, 'S气味': 592, 'E医学耗材': 793, 'S医学耗材': 794}

start_end_idx2str=dict(zip(list(start_end_str2idx.values()),list(start_end_str2idx.keys())))
#local_test=False # for test ,rule out invalid end, 症状结构化


def get_invalid_end(startid_ll): # 根据当前start id   最新的start id 找到无效的endid
    global start_end_idx2str, start_end_str2idx
    if startid_ll==[]:
        ll=[]
        # 如果没有START 所有END 都不合法
        for w, idx in start_end_str2idx.items():
            if w.startswith('E'):
                ll.append(start_end_str2idx[w])
        return ll
    ########
    startid=startid_ll[-1]
    start_w=start_end_idx2str[startid].strip('S')
    ll=[]
    for w,idx in start_end_str2idx.items():
        if start_w in w:
            continue # valid start end
        ###
        if w.startswith('E'):
            ll.append(start_end_str2idx[w])
    return ll

def is_start(this_step_id):
    global start_end_idx2str, start_end_str2idx
    if this_step_id in start_end_idx2str and start_end_idx2str[this_step_id].startswith('S'):
        return True
    else:
        return False

def is_end(this_step_id):
    global start_end_idx2str, start_end_str2idx
    if this_step_id in start_end_idx2str and start_end_idx2str[this_step_id].startswith('E'):
        return True
    else:
        return False

def initiate_end_start_by_beamsz(beam_size):
    real_start_stack = {}
    invalid_end_list={}
    for b in range(beam_size):
        real_start_stack[b]=[]
        invalid_end_list[b]=[]
    return real_start_stack,invalid_end_list
def make_neg_inf(beam_size,lprobs,invalid_end_list):
    for b_ in range(beam_size):  # yr
        lprobs[b_, invalid_end_list[b_]] = -math.inf
    return lprobs



