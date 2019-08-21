import ipywidgets as widgets
import matplotlib.pyplot as plt
import pandas
import IPython

def create_tsne_graph(error_detector, graph_out):
    """
    This function creates the matplotlib graph as well as any related widgets and returns it.
    """
    column_dropdown = widgets.Dropdown(
        options=error_detector.model.d.dataframe.columns,
        value=error_detector.model.d.dataframe.columns[0],
        description='Column: '
    )

    @graph_out.capture(clear_output=True)
    def onpick(event):
        """
        This function defines what happens when a point on the graph is clicked.
        """
        with graph_out:
            point = event.artist
            column = point.axes.get_title().split()[2]
            j = error_detector.model.d.dataframe.columns.get_loc(column)
            ind = event.ind[0]
            data = event.artist.get_offsets()
            xdata, ydata = data[ind, :]
            i = error_detector.model.models[j].tolist().index([xdata, ydata])
            cell = error_detector.model.d.dataframe.iat[i, j]
            row = pandas.DataFrame(data=[error_detector.model.d.dataframe.iloc[i, :]], columns=error_detector.model.d.dataframe.columns)
            print("Selected cell:\n{}\n\nFound in Tuple:".format(cell))
            IPython.display.display(row.style.hide_index().apply(
                lambda x: [
                    "background-color: #f2f2f2" if row.columns.get_loc(x.name) % 2 == 0 else "background-color: #d9d9d"]
            ))

            error_strats = {"dboost": [],
                            "katara": [],
                            "fdchecker": [],
                            "regex": []}
            for strategy in error_detector.model.d.cells_strategies[(i, j)]:
                for error_strat in error_strats:
                    if error_strat in strategy:
                        error_strats[error_strat].append(format_configuration_name(strategy))

            print("\n\n\nThese tools have detected this cell as an error:")
            number_of_configs = [len(configs) for configs in error_strats.values()]
            temp_data = {"Tool": list(error_strats.keys()), "Number of tool configurations": number_of_configs}
            tool_info_out = widgets.Output()
            with tool_info_out:
                IPython.display.display(pandas.DataFrame(temp_data).style.hide_index())

            configuration_list_boxes = []
            for tool in error_strats:
                if len(error_strats[tool]) == 0:
                    continue
                configuration_list = [widgets.Label(x) for x in error_strats[tool]]
                configuration_list_boxes.append(widgets.VBox(configuration_list))

            accord = widgets.Accordion(children=configuration_list_boxes)
            accord.selected_index = None
            i = 0
            for tool in error_strats:
                if len(error_strats[tool]) == 0:
                    continue
                accord.set_title(i, tool + " configurations")
                i += 1
            tool_info_container = widgets.VBox([tool_info_out, accord])

            IPython.display.display(tool_info_container)

    def displayCol(column):
        """
        Refreshes the matplotlib graph with the new column stated.
        """
        plt.close()
        fig, ax = plt.subplots()
        j = error_detector.model.d.dataframe.columns.get_loc(column)
        clean_xy = [data for i, data in enumerate(error_detector.model.models[j]) if (i, j) not in error_detector.model.correction_dictionary]
        dirty_xy = [data for i, data in enumerate(error_detector.model.models[j]) if (i, j) in error_detector.model.correction_dictionary]
        g = plt.scatter([c[0] for c in clean_xy], [c[1] for c in clean_xy], c='g', picker=5)
        r = plt.scatter([c[0] for c in dirty_xy], [c[1] for c in dirty_xy], c='r', picker=5)
        plt.title("{} Clusters ".format(error_detector.model.k - 1) + error_detector.model.d.dataframe.columns[j])
        fig.canvas.mpl_connect('pick_event', onpick)
        plt.legend(
            (g, r),
            ("Not errors", "Errors")
        )
        plt.show()

    return widgets.interactive(displayCol, column=column_dropdown)


def format_configuration_name(strategy):
    """
    Helper function to format configuration names for display.
    """
    raw_strategy = strategy.split(',', 1)[1]
    formatted_strategy = "".join(c for c in raw_strategy if c not in "[]")
    if 'katara' in strategy:
        formatted_strategy = formatted_strategy.split('/')[-1]
        if '"' == formatted_strategy[-1]:
            formatted_strategy = formatted_strategy[:-1]
    return formatted_strategy
