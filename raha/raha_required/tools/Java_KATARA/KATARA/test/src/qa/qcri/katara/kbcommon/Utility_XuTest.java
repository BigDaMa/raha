package qa.qcri.katara.kbcommon;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.util.*;

import qa.qcri.katara.dbcommon.Table;
import qa.qcri.katara.kbcommon.KBReader;
import qa.qcri.katara.kbcommon.KnowledgeDatabaseConfig;
import qa.qcri.katara.kbcommon.PatternDiscoveryException;
import qa.qcri.katara.kbcommon.util.StringPair;

import java.io.*;
public class Utility_XuTest {

	private static String kb = null;
	private static String kbStats = null;
	private static String minfreq = "0.1";
	
	public static void main(String[] args) throws Exception {
		
		
		int kbType = 2; 
		if(kbType == 1){
			kb = "/home/x4chu/Documents/KBs/simpleyago";
			kbStats = "/home/x4chu/Documents/KBs/simpleyagoStats/";
		} else if(kbType == 2){
			kb = "/home/x4chu/Documents/KBs/yagodata";
			kbStats = "/home/x4chu/Documents/KBs/yagodataStats";
		} else if(kbType == 3){
			kb = "/home/x4chu/Documents/KBs/dbpediadata";
			kbStats = "/home/x4chu/Documents/KBs/dbpediadataStats/";
		}
		KnowledgeDatabaseConfig.setDataDirectoryBase(kb);
		KnowledgeDatabaseConfig.KBStatsDirectoryBase = kbStats;
		KnowledgeDatabaseConfig.frequentPercentage = Double.valueOf(minfreq);
		System.out.println("Using KB " + KnowledgeDatabaseConfig.getKBName());
		
		//Doing the experiment
		String dir = "/home/x4chu/Documents/katara_working_table/EX/tablesForAnnotation/relationGT";
		test();
		
		

		String rdb = "/home/x4chu/Documents/katara_working_table/EX/tablesForAnnotation/relationGT/18";	
		//perform_single_exp(rdb);

	}
	private static void test() throws Exception
	{
		/*String type1 = "http://yago-knowledge.org/resource/wikicategory_European_countries";
		String type2 = 
				"http://yago-knowledge.org/resource/wordnet_physical_entity_100001930";
		//"http://yago-knowledge.org/resource/wordnet_country_108544813";
		
		String type3 = "http://yago-knowledge.org/resource/wikicategory_Capitals_in_Europe";
		String type4 = "http://yago-knowledge.org/resource/wordnet_capital_108518505";
		
		String rel = "http://yago-knowledge.org/resource/hasCapital";
		*/
		
		
		
		KBReader reader = new KBReader();
		/*int a = reader.getMinTypeSteps(type1,type2);
		System.out.println(a);
		System.out.println(reader.isSuperClassOf(type2, type1));
		
		int b = reader.getMinTypeSteps(type3, type4);
		System.out.println(b);

		System.out.println(reader.isSuperClassOf(type4, type3));
		
		Set<StringPair> sp = reader.getSubjectObjectGivenRel("http://yago-knowledge.org/resource/placedIn");
		System.out.println("placed in has" + sp.size());*/
		
		/*String rel = "http://yago-knowledge.org/resource/actedIn";;
		String obj = "http://yago-knowledge.org/resource/Vendetta_for_the_Saint";
		
		Set<String> subs = reader.getSubjectEntitiesGivenRelAndObject(rel, obj);*/
		
		String type1 = "http://yago-knowledge.org/resource/Spalding_Gray";

		String type2 = 	"http://yago-knowledge.org/resource/Swimming_to_Cambodia";
		Set<String> labels = reader.getLabels(type1);
		Set<String> label2 = reader.getLabels(type2);
		
		/*String type = "http://yago-knowledge.org/resource/wordnet_seat_108647945";
		long numDirectEn = reader.getEntities_Direct(type).size();
		System.out.println(" Type: " + type + " has num diret entities: " + numDirectEn  ) ;		
		long numEntities = reader.getEntities(type).size();
		System.out.println(" Type: " + type + " has num entities: " + numEntities ) ;		
				*/
		
		
		
		reader.close();
	}
}
