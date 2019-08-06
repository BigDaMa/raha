package qa.qcri.katara.crowdcommon.dao;

import java.util.ArrayList;
import java.util.List;

import qa.qcri.katara.dbcommon.Cell;
import qa.qcri.katara.dbcommon.Tuple;
import qa.qcri.katara.kbcommon.util.Pair;

public class MultipleChoiceQuestionContent extends QuestionContent {
	protected ArrayList<Pair<String, String>> candidates = null;

	protected List<ArrayList<String>> rows = new ArrayList<>();

	protected List<String> heads;

	public MultipleChoiceQuestionContent(List<String> heads,
			ArrayList<Pair<String, String>> candidateAnswers,
			List<Tuple> tuples) {
		super();

		this.heads = heads;

		this.candidates = candidateAnswers;

		for (Tuple t : tuples) {
			ArrayList<String> row = new ArrayList<>();
			for (Cell c : t.getTuple()) {
				row.add(c.getValue());
			}

			rows.add(row);
		}
	}
}
