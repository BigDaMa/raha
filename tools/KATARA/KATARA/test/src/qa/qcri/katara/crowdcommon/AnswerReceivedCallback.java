package qa.qcri.katara.crowdcommon;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

import org.apache.log4j.Logger;

import qa.qcri.crowdservice.crowdcommon.dao.Answer;
import qa.qcri.crowdservice.crowdcommon.dao.Question;
import qa.qcri.katara.crowdcommon.dao.*;

public class AnswerReceivedCallback {

	private static Logger log = Logger.getLogger(AnswerReceivedCallback.class);

	private static final int ASSIGNMENTS_PER_QUESTION = 5;

	/**
	 * TODO: move actual logic to related modules
	 * 
	 * @param question
	 * @param answer
	 *            the answer just received
	 * @param allAnswers
	 *            all the answers to this question, <b>INCLUDING</b> the current
	 *            one
	 * @param params
	 *            other parameters
	 */
	public static void handleAnswer(Question question, Answer currentAnswer,
			Collection<Answer> allAnswers, Map<String, String> params,
			ClassLoader classLoader) {
		log.debug("the parameters of answer handling are as follows");

		for (String key : params.keySet()) {
			log.debug("the parameter key is " + key + " value is "
					+ params.get(key));
		}

		Class<?> datavalidationAnswerResolverClass = null;
		Method resolveAnswerForDataValidationTypeQuestionMethod = null;
		// rdb file path
		String rdbFilePath = params.get("rdbfilepath");
		// interim result file path
		String interimResultFilePath = params.get("interimresultfilepath");
		// pattern file path
		String patternFilePath = params.get("patternfilepath");
		// pattern, tuple, question relationship file path
		String patternRelationQuestionFilePath = params
				.get("patternrelationquestionfilepath");
		// final result file path
		String finalResolveFilePath = params.get("finalresolvefilepath");
		// answer file path
		String answerFilePath = params.get("answerfilepath");
		try {

			datavalidationAnswerResolverClass = classLoader
					.loadClass("qa.qcri.katara.datavalidation.AnswerResolver");
			Class[] paramClasses = { question.getClass(),
					currentAnswer.getClass(), java.util.Collection.class,
					String.class, String.class, String.class, String.class,
					String.class, String.class };
			Method[] methods = datavalidationAnswerResolverClass
					.getDeclaredMethods();
			resolveAnswerForDataValidationTypeQuestionMethod = datavalidationAnswerResolverClass
					.getDeclaredMethod(
							"resolveAnswerForDataValidationTypeQuestion",
							paramClasses);
			String questionType = question.getType();
			switch (questionType) {
			case Question.DATAVALIDATIONBOOLEANQUESTION:
				resolveAnswerForDataValidationTypeQuestionMethod.invoke(null,
						new Object[] { question, currentAnswer, allAnswers,
								patternFilePath,
								patternRelationQuestionFilePath, rdbFilePath,
								interimResultFilePath, finalResolveFilePath,
								answerFilePath });
				break;
			case Question.DATAVALIDATIONRELATIONSHIPBOOLEANQUESTION:
				resolveAnswerForDataValidationTypeQuestionMethod.invoke(null,
						new Object[] { question, currentAnswer, allAnswers,
								patternFilePath,
								patternRelationQuestionFilePath, rdbFilePath,
								interimResultFilePath, finalResolveFilePath,
								answerFilePath });
				break;
			}

		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		} catch (NoSuchMethodException e) {
			e.printStackTrace();
		} catch (SecurityException e) {
			e.printStackTrace();
		} catch (IllegalAccessException e) {
			e.printStackTrace();
		} catch (IllegalArgumentException e) {
			e.printStackTrace();
		} catch (InvocationTargetException e) {
			e.printStackTrace();
		}

		// Finished collecting answers: resolve the majority vote
		if (allAnswers.size() == ASSIGNMENTS_PER_QUESTION) {
			HashMap<String, Integer> counts = new HashMap<>();

			for (Answer answer : allAnswers) {
				String a = answer.getContent();
				if (counts.containsKey(a)) {
					counts.put(a, counts.get(a) + 1);
				} else {
					counts.put(a, 1);
				}
			}

			// Find the answers with highest votes
			String bestAnswer = null;
			double maxScore = 0;

			for (String a : counts.keySet()) {
				if (counts.get(a) > maxScore) {
					maxScore = counts.get(a);
					bestAnswer = a;
				}
			}

			if (bestAnswer == null) {
				// no answer received...(e.g. all are "I don't know")

			}

			// See if absolute majority
			if (counts.get(bestAnswer) * 2 > ASSIGNMENTS_PER_QUESTION) {
				// Absolute majority
			} else {
				// maybe we want to ask further
			}

			//
		}
	}

}
