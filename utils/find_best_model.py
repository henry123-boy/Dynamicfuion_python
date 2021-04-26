import os


def find_best_model_name(model_dirname, data_version, verbose=False):
    from tensorflow.python.summary.summary_iterator import summary_iterator

    import tensorflow.python.util.deprecation as deprecation
    deprecation._PRINT_DEPRECATION_WARNINGS = False

    events_file_dir = "/data/nonrigid/training/tf_runs/deformdata/{}/{}".format(model_dirname, data_version)

    events_files = os.listdir(events_file_dir)
    assert len(events_files) == 1
    events_file = os.path.join(events_file_dir, events_files[0])

    if not os.path.exists(events_file):
        print()
        raise Exception("File does not exist! Exiting.")

    best_step = 0
    best_error = float('Inf')

    for e in summary_iterator(events_file):

        for v in e.summary.value:
            if not v.tag == "Metrics/EPE_3D":
                continue

            if v.simple_value < best_error:
                best_error = v.simple_value
                best_step = e.step

    if verbose:
        print("Best step {} at step {}".format(best_error, best_step))

    # Find model name based on step
    model_dir = "/data/nonrigid/training/models/deformdata/{}".format(model_dirname)

    best_model_name = None
    for mn in os.listdir(model_dir):
        model_step = mn.split("_")[3]

        if model_step == str(best_step):
            assert best_model_name == None
            best_model_name = mn

    return os.path.splitext(best_model_name)[0]
