import numpy
import math
import pickle
import sklearn
import os
import pandas


class DetectionModel:

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
        self.classification_model = classification_model
        self.labels_per_cluster = {}
        self.si = None

    def detect_errors(self):
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

    def create_labeled_tuple(self):
        self.labels_per_cluster = {}
        for j in range(self.d.dataframe.shape[1]):
            for c in self.clusters_k_j_c_ce[self.k][j]:
                self.labels_per_cluster[(j, c)] = {cell: self.labeled_cells[cell] for cell in
                                              self.clusters_k_j_c_ce[self.k][j][c]
                                              if cell[0] in self.labeled_tuples}
        tuple_score = {i: 0.0 for i in range(self.d.dataframe.shape[0]) if i not in self.labeled_tuples}
        for i in tuple_score:
            score = 0.0
            for j in range(self.d.dataframe.shape[1]):
                if not self.clusters_k_j_c_ce[self.k][j]:
                    continue
                cell = (i, j)
                c = self.cells_clusters_k_j_ce[self.k][j][cell]
                score += math.exp(-len(self.labels_per_cluster[(j, c)]))
            tuple_score[i] = math.exp(score)
        sum_tuple_score = sum(tuple_score.values())
        p_tuple_score = [float(v) / sum_tuple_score for v in tuple_score.values()]
        self.si = numpy.random.choice(list(tuple_score.keys()), 1, p=p_tuple_score)[0]
        # si, score = max(tuple_score.iteritems(), key=operator.itemgetter(1))
        self.labeled_tuples[self.si] = tuple_score[self.si]
        sampled_tuple = pandas.DataFrame(data=[self.d.dataframe.iloc[self.si, :]], columns=self.d.dataframe.columns)

        return sampled_tuple

    def process_input(self, check_answers):
        for j in range(self.d.dataframe.shape[1]):
            cell = (self.si, j)
            value = self.d.dataframe.iloc[cell]
            self.labeled_cells[cell] = check_answers[j]
            if cell in self.cells_clusters_k_j_ce[self.k][j]:
                c = self.cells_clusters_k_j_ce[self.k][j][cell]
                self.labels_per_cluster[(j, c)][cell] = self.labeled_cells[cell]

        extended_labeled_cells = dict(self.labeled_cells)
        for j in self.clusters_k_j_c_ce[self.k]:
            for c in self.clusters_k_j_c_ce[self.k][j]:
                if len(self.labels_per_cluster[(j, c)]) > 0 and \
                        sum(self.labels_per_cluster[(j, c)].values()) in [0, len(self.labels_per_cluster[(j, c)])]:
                    for cell in self.clusters_k_j_c_ce[self.k][j][c]:
                        extended_labeled_cells[cell] = self.labels_per_cluster[(j, c)].values()[0]

        self.create_correction_dict(extended_labeled_cells)
        self.k += 1

    def create_TSNE(self):
        print("Building TSNE Models...")
        for j in range(len(self.fv)):
            print("  Column {}...".format(j))
            fv_list = [self.fv[j][(i, j)] for i in range(len(self.fv[j]))]
            tsne_model = sklearn.manifold.TSNE(n_components=2, n_iter=400).fit_transform(fv_list)
            self.models.append(tsne_model)