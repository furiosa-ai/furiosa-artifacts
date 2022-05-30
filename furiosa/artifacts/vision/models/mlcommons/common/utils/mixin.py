from furiosa.runtime import session


class SessionMixin(object):
    def open_session(self):
        self.sess = session.create(self.model)

    def close_session(self):
        if not (self.sess is None):
            self.sess.close()

    def __enter__(self):
        self.open_session()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_session()

    def __del__(self):
        self.close_session()
