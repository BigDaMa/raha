package qa.qcri.katara.crowdcommon.dao;

import java.util.ArrayList;
import java.util.List;

import qa.qcri.katara.dbcommon.Tuple;
import qa.qcri.katara.kbcommon.util.Pair;

public class RelMultipleChoiceQuestionContent extends
		MultipleChoiceQuestionContent {
	protected int highlightedColumn1;
	protected int highlightedColumn2;

	public RelMultipleChoiceQuestionContent(List<String> heads,
			ArrayList<Pair<String, String>> candidateAnswers,
			List<Tuple> tuples, int requiredColIdx1, int requiredColIdx2) {
		super(heads, candidateAnswers, tuples);
		this.highlightedColumn1 = requiredColIdx1;
		this.highlightedColumn2 = requiredColIdx2;
		type = "PATTERNVALIDATIONTYPETHREE";
	}
}
