# Rolling-origin validation folds.
#
# In each fold we stand at the origin week T (the last week we know):
#   - train on every example whose TARGET week is T or earlier
#     (so the model never sees orders after week T)
#   - test on weeks T+1 ... T+10, all forecast from origin T (so h = week - T)
#
#   fold 1: train target weeks 1-115, test 116-125
#   fold 2: train target weeks 1-125, test 126-135
#   fold 3: train target weeks 1-135, test 136-145

ORIGINS = [115, 125, 135]
HORIZON = 10


def split_fold(data, origin):
    """Return (train, test) for one fold. `data` is the table from features.build_dataset."""
    train = data[data['week'] <= origin]

    test = data[(data['origin_week'] == origin) &
                (data['week'] > origin) &
                (data['week'] <= origin + HORIZON)]

    return train, test
