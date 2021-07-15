from tensorboardX import SummaryWriter


class Logger(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        self.writer.add_scalar(tag, value, step)
        self.writer.flush()

    def list_of_scalars_summary(self, tag_value_pairs, step, average=True):
        """Log scalar variables."""
        for tag, value in tag_value_pairs:
            if type(value) == dict:
                if average:
                    value['average'] = sum(value.values()) / len(value)
                self.writer.add_scalars(tag, value, step)
            else:
                self.writer.add_scalar(tag, value, step)
        self.writer.flush()
