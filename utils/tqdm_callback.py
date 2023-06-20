from tqdm.auto import tqdm as std_tqdm


class TqdmCallback(std_tqdm):
    callback = None

    def __init__(self, *args, **kwargs) -> None:
        self.callback = kwargs.pop('callback')
        super().__init__(*args, **kwargs)

    def update(self, n=1):
        displayed = super(TqdmCallback, self).update(n)
        if displayed:
            if self.callback:
                self.callback(self.format_dict['n'])
        return displayed
