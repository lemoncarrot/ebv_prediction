"""EBV prediction from H&E whole-slide images.

Four stages, each its own subpackage:

    preprocessing   segment tissue and tile it into level-0 patch coordinates
    embedding       read those patches and encode them with UNI2-h
    mil             train a MIL classifier on the per-slide embedding bags
    validation      check the encoder and score trained runs

`metadata` holds the label-table builders shared across stages.
"""
