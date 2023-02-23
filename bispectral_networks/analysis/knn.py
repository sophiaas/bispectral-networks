import scipy
import numpy as np


def analyze_dist(dist, dataset, top_n=100):
    breakdown = np.zeros((dist.shape[0], 2))
    total = 0
    for i, row in enumerate(dist):
        i_label = dataset.labels[i]
        top_idxs = np.argsort(row)[:top_n]
        # Include ties
        maxval = row[top_idxs[-1]]
        top_idxs = np.where(row <= maxval)[0]
        for j in top_idxs:
            if i == j:
                continue
            else:
                j_label = dataset.labels[j]
                if j_label == i_label:
                    # Same orbit
                    breakdown[i, 0] += 1
                else:
                    # Other
                    breakdown[i, 1] += 1

    breakdown_mean = breakdown.sum(axis=0) / (breakdown.shape[0])
    breakdown_percent = breakdown_mean / breakdown_mean.sum()
    return breakdown, breakdown_percent



def knn_analysis(model, dataset, n):
    model.eval()
    output, _ = model.forward(dataset.data.float())
    output = output.detach().cpu()
    output_dist = scipy.spatial.distance_matrix(output, output)
    nn_dist, nn_dist_mean = analyze_dist(output_dist, dataset, n)
    return output, output_dist, nn_dist_mean
