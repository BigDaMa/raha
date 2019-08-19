import ipywidgets as widgets
import sklearn

class StartButton(widgets.Button):

    def __init__(self, error_detector, graph_check):
        super().__init__(description='Start')

        def start_process(event):
            error_detector.model.show_graphs = graph_check.value
            error_detector.out.clear_output()
            if error_detector.model.show_graphs:
                with error_detector.out:
                    error_detector.model.create_TSNE()
            error_detector.out.clear_output()
            with error_detector.out:
                error_detector.label_tuple()

        self.on_click(start_process)
