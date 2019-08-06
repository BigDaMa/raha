package qa.qcri.katara.kbcommon;

public interface IKeywordExtractor {

	public String getTypeKeyWordFromURI(String URI);

	public String getTypeURIByKeyWord(String keyWord);

	public String getRelationshipKeyWordFromURI(String URI);

	public String getRelationshipURIFromKeyWord(String keyWord);
}