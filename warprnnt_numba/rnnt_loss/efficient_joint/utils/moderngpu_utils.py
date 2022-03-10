import torch

"""
// LoadBalanceSearch is a special vectorized sorted search. Consider `bCount`
// objects that generate a variable number of work items, with `aCount` work
// items in total. The caller computes an exclusive scan of the work-item counts
// into `b`.

// `indices` has `aCount` outputs. `indices[i]` is the index of the 
// object that generated the i'th work item.
// Eg:
// work-item counts:  2,  5,  3,  0,  1.
// scan counts:       0,  2,  7, 10, 10.   `aCount` = 11.
// 
// LoadBalanceSearch computes the upper-bound of counting_iterator<int>(0) with
// the scan of the work-item counts and subtracts 1:
// LBS: 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 4.

// This is equivalent to expanding the index of each object by the object's
// work-item count.
"""


def load_balance_search_cpu(aCount: int, b: torch.Tensor, bCount: int, indices: torch.Tensor):
    ai = 0
    bi = 0
    while ai < aCount or bi < bCount:
        if bi >= bCount:
            p = True
        elif ai >= aCount:
            p = False
        else:
            p = ai < b[bi]  # // aKey < bKey is upper - bound condition.

        if p:
            indices[ai] = bi - 1
            ai += 1
        else:
            bi += 1


def row_splits_to_row_ids_cpu(row_splits: torch.Tensor, num_elements: int = -1):
    num_rows = row_splits.size(0) - 1
    device = row_splits.device
    row_splits = row_splits.cpu()

    if num_elements < 0:
        num_elements = row_splits[num_rows]

    row_ids = torch.empty(num_elements, dtype=row_splits.dtype, device=row_splits.device)
    load_balance_search_cpu(aCount=num_elements, b=row_splits, bCount=num_rows, indices=row_ids)

    row_ids = row_ids.to(device=device)
    return row_ids
