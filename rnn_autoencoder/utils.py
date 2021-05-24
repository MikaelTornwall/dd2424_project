import time
import math


def search_episode(data, search_term, spotify=True):
    results = []
    for i, row in data.iterrows():        
        if spotify:            
            # if search_term in row['episode_desc']:                
            if search_term in row['body']:                
                results.append(row)
                
        else:
            if search_term in row['summary']:
                results.append(row)

    return results


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))
