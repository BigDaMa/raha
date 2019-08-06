package qa.qcri.katara.kbcommon;

import java.io.File;

import org.apache.jena.atlas.lib.StrUtils;

public class KnowledgeDatabaseConfig {

	public static double frequentPercentage = 0.4;

	public static int maxLength = 3;
	public static int maxMatches = -1;// -1 dictats exact matches

	public static int sampling = 0;
	public static final String prefixes = StrUtils.strjoin("\n", new String[] {
			"PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>",
			"PREFIX rdfs:    <http://www.w3.org/2000/01/rdf-schema#>",
			"PREFIX pf:     <http://jena.hpl.hp.com/ARQ/property#>",
			"PREFIX owl: <http://www.w3.org/2002/07/owl#>",
			"PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>",
			"PREFIX skos: <http://www.w3.org/2004/02/skos/core#>"
			})
			+ "\n";

	public static String defaultNS;

	public static final String ROOT_CLASS = "http://www.w3.org/2002/07/owl#Thing";

	public static String dataDirectoryBase = "";

	public static String languageTag;
	
	public static String KBStatsDirectoryBase = "";
	

	public static void setDataDirectoryBase(String dataDirectoryBase) {
		KnowledgeDatabaseConfig.dataDirectoryBase = dataDirectoryBase;
		
		if(dataDirectoryBase.toLowerCase().contains("yago"))
		{
			languageTag = "@eng";
			defaultNS = "http://yago-knowledge.org/resource/";
		}
		else if(dataDirectoryBase.toLowerCase().contains("dbpedia"))
		{
			languageTag = "@en";
			defaultNS = "http://dbpedia.org/resource/";
			
			System.out.println("defaultNS:"+ defaultNS);
		}else if (dataDirectoryBase.toLowerCase().contains("imdb")) {
			languageTag = "";
			defaultNS = "http://data.linkedmdb.org/";
		}
	}
	
	public static String getKBName(){
		
		return dataDirectoryBase.substring( dataDirectoryBase.lastIndexOf("/")+1);
	}
	

	public static void setSampling(int sampling) {
		KnowledgeDatabaseConfig.sampling = sampling;
	}

	public static String getDataDirectory() throws PatternDiscoveryException {
		if ("".equals(dataDirectoryBase)) {
			throw new PatternDiscoveryException("dataDirectoryBase is empty");
		}
		return dataDirectoryBase + File.separatorChar + "data"
				+ File.separatorChar;
	}

	public static String getIndexDirectory() throws PatternDiscoveryException {
		if ("".equals(dataDirectoryBase)) {
			throw new PatternDiscoveryException("dataDirectoryBase is empty");
		}
		return dataDirectoryBase + File.separatorChar + "index"
				+ File.separatorChar;
	}

	public static String getSourceDirectory() throws PatternDiscoveryException {
		if ("".equals(dataDirectoryBase)) {
			System.err.println("dataDirectoryBase is empty!");
			throw new PatternDiscoveryException("dataDirectoryBase is empty");
		}
		return dataDirectoryBase + File.separatorChar + "ttl"
				+ File.separatorChar;
	}
	
	
	public static boolean interestedTypes(String candidateType){
		
		//KB Specific
		if(KnowledgeDatabaseConfig.dataDirectoryBase.toLowerCase().contains("yago")){
			//No action for yago
			
			if(candidateType.startsWith("http://yago-knowledge.org/resource/wikicategory_SAM_Coup"))
				return false;
			if(!candidateType.startsWith("http://yago-knowledge.org/resource/")){
				return false;
			}else{
				return true;
			}
		} else if(KnowledgeDatabaseConfig.dataDirectoryBase.toLowerCase().contains("dbpedia")){
			//dbpedia
			if(!candidateType.startsWith("http://dbpedia.org/")){
				return false;
			}else{
				return true;
			}
		}else{
			return true;
		}
	}
	
	public static boolean interestedRelationships(String candidateRel){
		if(KnowledgeDatabaseConfig.dataDirectoryBase.toLowerCase().contains("yago")){
			//yago
			if(candidateRel.equals("http://yago-knowledge.org/resource/placedIn")
					|| candidateRel.equals("http://yago-knowledge.org/resource/wikicategory_Subject?object?verb_languages"))
				return false;
			
			if(!candidateRel.startsWith("http://yago-knowledge.org/resource/"))
				return false;
			else if (candidateRel.equals("http://yago-knowledge.org/resource/linksTo")
				|| candidateRel.equals("http://yago-knowledge.org/resource/isPreferredMeaningOf")) {
				return false;
			}else{
				return true;
			}
		} else if(KnowledgeDatabaseConfig.dataDirectoryBase.toLowerCase().contains("dbpedia")){
			//dbpedia
			if(!candidateRel.startsWith("http://dbpedia.org/property/")){
				return false;	
			}else{
				return true;
			}
		}else{
			return true;
		}
	}
}
