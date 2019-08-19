import ipywidgets as widgets
import IPython.display
from raha.tools.error_detector import rahaModel
from raha.tools.error_detector.rwidgets import start_button, submit_button, column_graph_displayer, another_tuple_button

class ErrorDetector:

    def __init__(self, labeling_budget, d, run_count, ed_folder_path, fv, classification_model):
        self.model = rahaModel.DetectionModel(labeling_budget, d, run_count, ed_folder_path, fv, classification_model)
        self.out = widgets.Output()

    def error_detection(self):
        self.model.detect_errors()

    def interactive_error_detection(self):
        IPython.display.clear_output()
        graph_check = widgets.Checkbox(description='Graphs')
        graphbox = widgets.VBox([widgets.Label('Shows interactive graphs (May take a while to start)'), graph_check])
        start = start_button.StartButton(self, graph_check)
        vbox = widgets.VBox([widgets.Label('Start the interactive process'), start])

        IPython.display.display(self.out)
        with self.out:
            IPython.display.display(vbox, graphbox)

    def label_tuple(self):
        sampled_tuple = self.model.create_labeled_tuple()
        with self.out:
            print("Label the dirty cells in the following sampled tuple.")
            checks = [widgets.Checkbox(description=column) for column in sampled_tuple]
            submit = submit_button.SubmitButton(self, checks)

            IPython.display.display(sampled_tuple.style.hide_index().apply(
                lambda x: ["background-color: #f2f2f2" if sampled_tuple.columns.get_loc(x.name) % 2 == 0 else "background-color: #d9d9d9"]))
            layout = widgets.VBox(checks)
            IPython.display.display(layout, submit)

    def display_graph(self):
        graph_out = widgets.Output(layout={'border': '1px solid black'})
        graph_out_container = widgets.Accordion(children=[graph_out])
        graph_out_container.set_title(0, "Display clicked cell information")
        with graph_out:
            print("Please click a point to view more information.")
        if self.model.show_graphs:
            w = column_graph_displayer.create_tsne_graph(self, graph_out)
        else:
            pc = len(self.model.correction_dictionary)/len(self.model.d.dataframe)
            w = widgets.Label("{0:.2f}% of the dataset has been classified as errors".format(pc))

        another_tuple = another_tuple_button.AnotherTupleButton(self, graph_out, w)

        print("\n")
        IPython.display.display(another_tuple)
        print("\n")
        IPython.display.display(w)
        if self.model.show_graphs:
            IPython.display.display(graph_out_container)






