package qa.qcri.katara.kbcommon;

public class DBPediaKeyWordExtractor implements IKeywordExtractor {

	@Override
	public String getTypeKeyWordFromURI(String URI) {
		return URI.replaceAll("http://dbpedia.org/property/", "").toLowerCase();
	}

	@Override
	public String getTypeURIByKeyWord(String keyWord) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public String getRelationshipKeyWordFromURI(String URI) {
		return URI.replaceAll("http://dbpedia.org/property/", "").toLowerCase();
	}

	@Override
	public String getRelationshipURIFromKeyWord(String keyWord) {
		// TODO Auto-generated method stub
		return null;
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}

}
