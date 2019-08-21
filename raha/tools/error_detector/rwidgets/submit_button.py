import ipywidgets as widgets


class SubmitButton(widgets.Button):
    """
    Button to submit a labeled tuple and continue the error detection process.
    """

    def __init__(self, error_detector, checks):
        super().__init__(description='Submit')

        def submit_click(event):
            error_detector.out.clear_output()

            check_answers = []
            for check in checks:
                if check.value is True:
                    check_answers.append(1)
                else:
                    check_answers.append(0)

            error_detector.model.process_input(check_answers)

            with error_detector.out:
                error_detector.display_graph()

        self.on_click(submit_click)


