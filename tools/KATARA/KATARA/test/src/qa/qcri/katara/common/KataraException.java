package qa.qcri.katara.common;

public class KataraException extends Exception {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public KataraException(String exceptionContent) {
		super(exceptionContent);
	}

	public KataraException(Exception exc) {
		super(exc);
	}

}
