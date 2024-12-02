def check2(feat,top_k=5):
    global cnt
    tin = perf_counter()
    similarities = [(labels[i], cosine(features[i],feat)) for i in range(len(features))]
    similarities = sorted(similarities,key=lambda x:x[1], reverse=True)[:top_k]
    top_labels = [x[0] for x in similarities]
    most_common_label,_ = Counter(top_labels).most_common(1)[0]
    if cnt%10==0:print(f"check 2 in {perf_counter()-tin:.2f}s")
    cnt+=1
    return most_common_label, similarities
