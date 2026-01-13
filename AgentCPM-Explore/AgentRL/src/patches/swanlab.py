import transformers.integrations.integration_utils

def rewrite_logs(d):
    new_d = {}
    eval_prefix = "eval_"
    eval_prefix_len = len(eval_prefix)
    test_prefix = "test_"
    test_prefix_len = len(test_prefix)
    for k, v in d.items():
        if k.startswith(eval_prefix):
            new_d["eval/" + k[eval_prefix_len:]] = v
        elif k.startswith(test_prefix):
            new_d["test/" + k[test_prefix_len:]] = v
        elif k.startswith("/"):
            new_d[k[1:]] = v
        else:
            new_d["train/" + k] = v
    return new_d
transformers.integrations.integration_utils.rewrite_logs = rewrite_logs
