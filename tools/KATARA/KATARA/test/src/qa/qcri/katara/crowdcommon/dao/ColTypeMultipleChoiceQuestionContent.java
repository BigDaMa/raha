package qa.qcri.katara.crowdcommon.dao;

import java.util.ArrayList;
import java.util.List;

import qa.qcri.katara.dbcommon.Tuple;
import qa.qcri.katara.kbcommon.util.Pair;

public class ColTypeMultipleChoiceQuestionContent extends
		MultipleChoiceQuestionContent {
	protected int highlightedColumn;

	public ColTypeMultipleChoiceQuestionContent(List<String> heads,
			ArrayList<Pair<String, String>> candidateAnswers,
			List<Tuple> tuples, int requiredColIdx) {
		super(heads, candidateAnswers, tuples);
		this.highlightedColumn = requiredColIdx;
		type = "PATTERNVALIDATIONTYPEONE";
	}
}
