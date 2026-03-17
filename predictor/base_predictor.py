class BasePredictor:

    def predict_next_state(self, z_t, skill_spec):
        raise NotImplementedError
