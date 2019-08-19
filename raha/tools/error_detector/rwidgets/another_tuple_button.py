import ipywidgets as widgets


class AnotherTupleButton(widgets.Button):

    def __init__(self, error_detector, graph_out, w):
        super().__init__(description='Label another tuple')

        def onclick(event):
            if error_detector.model.k == error_detector.model.clustering_range[-1]:
                with error_detector.out:
                    print("Sorry, but the labeling budget has been reached.")
            else:
                graph_out.clear_output()
                # graph_out.close()
                error_detector.out.clear_output()
                w.close()
                with error_detector.out:
                    error_detector.label_tuple()

        self.on_click(onclick)
