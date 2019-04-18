package qa.qcri.katara.kbcommon;

/**
 * 
 * @author Yin Ye
 *
 */
public class PatternDiscoveryException extends Exception{

	private static final long serialVersionUID = 1L;

	private String exceptionContent;
	
	public PatternDiscoveryException(String message){
		exceptionContent = message;
	}
	
	
	public String getExceptionContent(){
		return exceptionContent;
	}	
}
