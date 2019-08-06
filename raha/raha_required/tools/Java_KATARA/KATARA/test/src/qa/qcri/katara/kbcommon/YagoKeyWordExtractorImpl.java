package qa.qcri.katara.kbcommon;

public class YagoKeyWordExtractorImpl implements IKeywordExtractor {

	@Override
	public String getTypeKeyWordFromURI(String URI) {
		return URI.replaceAll("http://yago-knowledge.org/resource/", "")
				.replaceAll("wordnet_", "").replaceAll("_\\d+", "")
				.replaceAll("wikicategory_", "").toLowerCase();
	}

	@Override
	public String getTypeURIByKeyWord(String keyWord) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public String getRelationshipKeyWordFromURI(String URI) {
		return URI.replaceAll("http://yago-knowledge.org/resource/", "");
	}

	@Override
	public String getRelationshipURIFromKeyWord(String keyWord) {
		// TODO Auto-generated method stub
		return null;
	}
}
