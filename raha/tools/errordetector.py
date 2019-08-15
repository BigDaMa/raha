import math
import pickle
import numpy
import sklearn
import os
import ipywidgets as widgets
import IPython.display
import pandas
import matplotlib.pyplot as plt

class ErrorDetector:

    def __init__(self, labeling_budget, d, run_count, ed_folder_path, fv, classification_model):
        self.labeling_budget = labeling_budget
        self.sampling_range = range(1, labeling_budget + 1)
        self.clustering_range = range(2, labeling_budget + 2)
        self.d = d
        self.clusters_k_j_c_ce = {k: {j: d.clusters_j_k_c_ce[j][k] for j in range(d.dataframe.shape[1])} for k in self.clustering_range}
        self.cells_clusters_k_j_ce = {k: {j: d.cells_clusters_j_k_ce[j][k] for j in range(d.dataframe.shape[1])} for k in self.clustering_range}
        self.aggregate_results = {s: [] for s in self.sampling_range}
        self.k = self.clustering_range[0]
        self.labeled_tuples = {}
        self.labeled_cells = {}
        self.interactive = False
        self.correction_dictionary = {}
        self.run_count = run_count
        self.ed_folder_path = ed_folder_path
        self.show_graphs = True
        self.models = []
        self.fv = fv
        self.out = widgets.Output()
        self.classification_model = classification_model

    def error_detection(self):
        for r in range(self.run_count):
            print("Run {}...".format(r))
            labeled_tuples = {}
            labeled_cells = {}
            for k in self.clusters_k_j_c_ce:
                labels_per_cluster = {}
                for j in range(self.d.dataframe.shape[1]):
                    for c in self.clusters_k_j_c_ce[k][j]:
                        labels_per_cluster[(j, c)] = {cell: labeled_cells[cell] for cell in self.clusters_k_j_c_ce[k][j][c] if cell[0] in labeled_tuples}
                tuple_score = {i: 0.0 for i in range(self.d.dataframe.shape[0]) if i not in labeled_tuples}
                for i in tuple_score:
                    score = 0.0
                    for j in range(self.d.dataframe.shape[1]):
                        if not self.clusters_k_j_c_ce[k][j]:
                            continue
                        cell = (i, j)
                        c = self.cells_clusters_k_j_ce[k][j][cell]
                        score += math.exp(-len(labels_per_cluster[(j, c)]))
                    tuple_score[i] = math.exp(score)
                sum_tuple_score = sum(tuple_score.values())
                p_tuple_score = [float(v) / sum_tuple_score for v in tuple_score.values()]
                si = numpy.random.choice(list(tuple_score.keys()), 1, p=p_tuple_score)[0]
                # si, score = max(tuple_score.iteritems(), key=operator.itemgetter(1))
                labeled_tuples[si] = tuple_score[si]
                for j in range(self.d.dataframe.shape[1]):
                    cell = (si, j)
                    labeled_cells[cell] = int(cell in self.d.actual_errors_dictionary)
                    if cell in self.cells_clusters_k_j_ce[k][j]:
                        c = self.cells_clusters_k_j_ce[k][j][cell]
                        labels_per_cluster[(j, c)][cell] = labeled_cells[cell]
                extended_labeled_cells = dict(labeled_cells)
                for j in self.clusters_k_j_c_ce[k]:
                    for c in self.clusters_k_j_c_ce[k][j]:
                        if len(labels_per_cluster[(j, c)]) > 0 and \
                                sum(labels_per_cluster[(j, c)].values()) in [0, len(labels_per_cluster[(j, c)])]:
                            for cell in self.clusters_k_j_c_ce[k][j][c]:
                                extended_labeled_cells[cell] = labels_per_cluster[(j, c)].values()[0]
                self.create_correction_dict(extended_labeled_cells)
                s = len(labeled_tuples)
                er = self.d.evaluate_data_cleaning(self.correction_dictionary)[:3]
                self.aggregate_results[s].append(er)
                pickle.dump(self.correction_dictionary, open(os.path.join(self.ed_folder_path, "results.dictionary"), "wb"))

        results_string = "\\addplot[error bars/.cd,y dir=both,y explicit] coordinates{(0,0.0)"
        for s in self.sampling_range:
            mean = numpy.mean(numpy.array(self.aggregate_results[s]), axis=0)
            std = numpy.std(numpy.array(self.aggregate_results[s]), axis=0)
            print("Raha on {}".format(self.d.name))
            print("Labeled Tuples Count = {}".format(s))
            print("Precision = {:.2f} +- {:.2f}".format(mean[0], std[0]))
            print("Recall = {:.2f} +- {:.2f}".format(mean[1], std[1]))
            print("F1 = {:.2f} +- {:.2f}".format(mean[2], std[2]))
            print("--------------------")
            results_string += "({},{:.2f})+-(0,{:.2f})".format(s, mean[2], std[2])
        results_string += "}; \\addlegendentry{Raha}"
        print(results_string)

    def interactive_error_detection(self):
        IPython.display.clear_output()
        graph_check = widgets.Checkbox(description='Graphs')
        graphbox = widgets.VBox([widgets.Label('Shows interactive graphs (May take a while to start)'), graph_check])
        start = widgets.Button(description='Start')
        vbox = widgets.VBox([widgets.Label('Start the interactive process'), start])
        def start_process(event):
            self.show_graphs = graph_check.value
            #start.layout.visibility = 'hidden'
            self.out.clear_output()
            if self.show_graphs:
                with self.out:
                    print("Building TSNE Models...")
                    for j in range(len(self.fv)):
                        print("  Column {}...".format(j))
                        fv_list = [self.fv[j][(i, j)] for i in range(len(self.fv[j]))]
                        model = sklearn.manifold.TSNE(n_components=2, n_iter=400).fit_transform(fv_list)
                        self.models.append(model)
            self.out.clear_output()
            with self.out:
                self.label_tuple()
        start.on_click(start_process)
        IPython.display.display(self.out)
        with self.out:
            IPython.display.display(vbox, graphbox)

    def label_tuple(self):
        label_out = widgets.Output()
        labels_per_cluster = {}
        for j in range(self.d.dataframe.shape[1]):
            for c in self.clusters_k_j_c_ce[self.k][j]:
                labels_per_cluster[(j, c)] = {cell: self.labeled_cells[cell] for cell in self.clusters_k_j_c_ce[self.k][j][c]
                                              if cell[0] in self.labeled_tuples}
        tuple_score = {i: 0.0 for i in range(self.d.dataframe.shape[0]) if i not in self.labeled_tuples}
        for i in tuple_score:
            score = 0.0
            for j in range(self.d.dataframe.shape[1]):
                if not self.clusters_k_j_c_ce[self.k][j]:
                    continue
                cell = (i, j)
                c = self.cells_clusters_k_j_ce[self.k][j][cell]
                score += math.exp(-len(labels_per_cluster[(j, c)]))
            tuple_score[i] = math.exp(score)
        sum_tuple_score = sum(tuple_score.values())
        p_tuple_score = [float(v) / sum_tuple_score for v in tuple_score.values()]
        si = numpy.random.choice(list(tuple_score.keys()), 1, p=p_tuple_score)[0]
        # si, score = max(tuple_score.iteritems(), key=operator.itemgetter(1))
        self.labeled_tuples[si] = tuple_score[si]
        IPython.display.display(label_out)
        with label_out:
            print("Label the dirty cells in the following sampled tuple.")
            sampled_tuple = pandas.DataFrame(data=[self.d.dataframe.iloc[si, :]], columns=self.d.dataframe.columns)
            sampled_tuple.style.hide_index()
            checks = [widgets.Checkbox(description=column) for column in sampled_tuple]
            submit = widgets.Button(description='Submit')

        def submit_click(event):
            check_answers = []
            for check in checks:
                if check.value is True:
                    check_answers.append(1)
                else:
                    check_answers.append(0)
                #check.close()
            label_out.clear_output()


            for j in range(self.d.dataframe.shape[1]):
                cell = (si, j)
                value = self.d.dataframe.iloc[cell]
                self.labeled_cells[cell] = check_answers[j]
                if cell in self.cells_clusters_k_j_ce[self.k][j]:
                    c = self.cells_clusters_k_j_ce[self.k][j][cell]
                    labels_per_cluster[(j, c)][cell] = self.labeled_cells[cell]


            extended_labeled_cells = dict(self.labeled_cells)
            for j in self.clusters_k_j_c_ce[self.k]:
                for c in self.clusters_k_j_c_ce[self.k][j]:
                    if len(labels_per_cluster[(j, c)]) > 0 and \
                            sum(labels_per_cluster[(j, c)].values()) in [0, len(labels_per_cluster[(j, c)])]:
                        for cell in self.clusters_k_j_c_ce[self.k][j][c]:
                            extended_labeled_cells[cell] = labels_per_cluster[(j, c)].values()[0]

            self.create_correction_dict(extended_labeled_cells)
            self.k += 1
            self.out.clear_output()

            with self.out:
                self.display_graph()

        submit.on_click(submit_click)

        with label_out:
            IPython.display.display(sampled_tuple.style.apply(
                lambda x: ["background-color: #f2f2f2" if sampled_tuple.columns.get_loc(x.name) % 2 == 0 else "background-color: #d9d9d9"]))
            layout = widgets.VBox(checks)
            IPython.display.display(layout, submit)

    def create_correction_dict(self, extended_labeled_cells):
        self.correction_dictionary = {}
        for j in range(self.d.dataframe.shape[1]):
            x_train = [self.d.fv[j][(i, j)] for i in range(self.d.dataframe.shape[0]) if (i, j) in extended_labeled_cells]
            y_train = [extended_labeled_cells[(i, j)] for i in range(self.d.dataframe.shape[0]) if
                       (i, j) in extended_labeled_cells]
            x_test = [self.d.fv[j][(i, j)] for i in range(self.d.dataframe.shape[0])]
            test_cells = [(i, j) for i in range(self.d.dataframe.shape[0])]
            if sum(y_train) == len(y_train):
                predicted_labels = len(test_cells) * [1]
            elif sum(y_train) == 0 or len(x_train[0]) == 0:
                predicted_labels = len(test_cells) * [0]
            else:
                if self.classification_model == "ABC":
                    classification_model = sklearn.ensemble.AdaBoostClassifier(n_estimators=100)
                if self.classification_model == "DTC":
                    classification_model = sklearn.tree.DecisionTreeClassifier(criterion="gini")
                if self.classification_model == "GBC":
                    classification_model = sklearn.ensemble.GradientBoostingClassifier(n_estimators=100)
                if self.classification_model == "GNB":
                    classification_model = sklearn.naive_bayes.GaussianNB()
                if self.classification_model == "KNC":
                    classification_model = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1)
                if self.classification_model == "SGDC":
                    classification_model = sklearn.linear_model.SGDClassifier(loss="hinge", penalty="l2")
                if self.classification_model == "SVC":
                    classification_model = sklearn.svm.SVC(kernel="sigmoid")
                classification_model.fit(x_train, y_train)
                predicted_labels = classification_model.predict(x_test)
            for index, pl in enumerate(predicted_labels):
                cell = test_cells[index]
                if (cell[0] in self.labeled_tuples and extended_labeled_cells[cell]) or \
                        (cell[0] not in self.labeled_tuples and pl):
                    self.correction_dictionary[cell] = "JUST A DUMMY VALUE"
        pickle.dump(self.correction_dictionary, open(os.path.join(self.ed_folder_path, "results.dictionary"), "wb"))

    def display_graph(self):
        graph_out = widgets.Output(layout={'border': '1px solid black'})
        with graph_out:
            print("Please click a point to view more information.")

        if self.show_graphs:
            column_dropdown = widgets.Dropdown(
                options=self.d.dataframe.columns,
                value=self.d.dataframe.columns[0],
                description='Column: '
            )
            @graph_out.capture(clear_output=True)
            def onpick(event):
                with graph_out:
                    point = event.artist
                    column = point.axes.get_title().split()[2]
                    j = self.d.dataframe.columns.get_loc(column)
                    ind = event.ind[0]
                    data = event.artist.get_offsets()
                    xdata, ydata = data[ind, :]
                    i = self.models[j].tolist().index([xdata, ydata])
                    cell = self.d.dataframe.iat[i, j]
                    row = self.d.dataframe.iloc[i, :]
                    print("Selected cell:\n{}\n\nFound in Tuple:\n".format(cell))
                    row = pandas.DataFrame(row)
                    row.style.hide_index()
                    IPython.display.display(row)

                    error_strats = {"dboost": 0,
                                    "katara": 0,
                                    "fdchecker": 0,
                                    "regex": 0}
                    for strategy in self.d.cells_strategies[(i,j)]:
                        for error_strat in error_strats:
                            if error_strat in strategy:
                                error_strats[error_strat] += 1

                    print("\nThese tools have detected this cell as an error:")
                    temp_data = {"Tool": list(error_strats.keys()), "Number of tool configurations": list(error_strats.values())}
                    IPython.display.display(pandas.DataFrame(temp_data))

            def displayCol(column):
                plt.close()
                fig, ax = plt.subplots()
                j = self.d.dataframe.columns.get_loc(column)
                clean_xy = [data for i, data in enumerate(self.models[j]) if (i, j) not in self.correction_dictionary]
                dirty_xy = [data for i, data in enumerate(self.models[j]) if (i, j) in self.correction_dictionary]
                g = plt.scatter([c[0] for c in clean_xy], [c[1] for c in clean_xy], c='g', picker=5)
                r = plt.scatter([c[0] for c in dirty_xy], [c[1] for c in dirty_xy], c='r', picker=5)
                plt.title("{} Clusters ".format(self.k -1) + self.d.dataframe.columns[j])
                fig.canvas.mpl_connect('pick_event', onpick)
                plt.legend(
                    (g ,r),
                    ("Not errors", "Errors")
                )
                plt.show()
                IPython.display.display(graph_out)
            w = widgets.interactive(displayCol, column=column_dropdown)
            IPython.display.display(w)
        else:
            pc = len(self.correction_dictionary)/len(self.d.dataframe)
            pc_label = widgets.Label("{0:.2f}% of the dataset has been classified as errors".format(pc))
            IPython.display.display(pc_label)

        another_tuple = widgets.Button(description='Label another tuple')

        def onclick(event):
            if self.k == self.clustering_range[-1]:
                with self.out:
                    print("Sorry, but the labeling budget has been reached.")
            else:
                graph_out.clear_output()
                #graph_out.close()
                self.out.clear_output()
                if self.show_graphs:
                    w.close()
                with self.out:
                    self.label_tuple()

        another_tuple.on_click(onclick)
        IPython.display.display(another_tuple)


