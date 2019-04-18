package simplied.katara;


import qa.qcri.katara.common.config.KataraConfigure;
import qa.qcri.katara.kbcommon.KBReader;
import qa.qcri.katara.kbcommon.KBStatsManager;
import qa.qcri.katara.kbcommon.KnowledgeDatabaseConfig;
import qa.qcri.katara.kbcommon.pattern.simple.TableSemantics;
import qa.qcri.katara.kbcommon.pattern.simple.TypeRelCoherence;
import qa.qcri.katara.kbcommon.util.StringPair;

import java.io.*;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;



public class RankJoinKBStats {

	
	
	boolean KBStatsComplete = false; //Whether or not we have built the complete stats for the KB
	public static String typeStatsPath = null;
	public static String relStatsPath = null;
	public static String typeRelCoherencePath = null;
	
	
	static Map<String, Integer> type2NumEntities = new HashMap<String,Integer>();
	static Map<String, Integer> rel2NumEntities = new HashMap<String,Integer>();
	static Map<TypeRelCoherence,Double> trcIndex = new HashMap<TypeRelCoherence,Double>();
	
	static {
		if(KnowledgeDatabaseConfig.KBStatsDirectoryBase.endsWith(File.separator))
		{
			typeRelCoherencePath = KnowledgeDatabaseConfig.KBStatsDirectoryBase + "typeRelCohe.inx";
			typeStatsPath = KnowledgeDatabaseConfig.KBStatsDirectoryBase + "types.inx";
			relStatsPath = KnowledgeDatabaseConfig.KBStatsDirectoryBase + "rel.inx";
		}
		else
		{
			typeRelCoherencePath = KnowledgeDatabaseConfig.KBStatsDirectoryBase + File.separator +  "typeRelCohe.inx";
			typeStatsPath = KnowledgeDatabaseConfig.KBStatsDirectoryBase + File.separator + "types.inx";
			relStatsPath = KnowledgeDatabaseConfig.KBStatsDirectoryBase + File.separator + "rel.inx";
		}
		
		
		//Init types stats
		BufferedReader in;
		try {
			in = new BufferedReader(new FileReader(typeStatsPath));
			String line = null;
			int typeIndex = -1;
			while((line = in.readLine()) != null)
			{
				typeIndex++;
				String[] splits = line.split(",");
				StringBuilder sb = new StringBuilder();
				for(int i = 0; i < splits.length -1 ; i++)
				{
					if(i == 0)
					{
						sb.append(splits[i]);
					}
					else 
					{
						sb.append(",");
						sb.append(splits[i]);
					}
				}
				String key = sb.toString();
				int count = Integer.valueOf(splits[splits.length - 1]);
				type2NumEntities.put(key,count);
			}
			in.close();
		} catch (IOException e2) {
			// TODO Auto-generated catch block
			e2.printStackTrace();
		}
		
		
		
		//Init rel stats
		try {
			in = new BufferedReader(new FileReader(relStatsPath));
			String line = null;
			while((line = in.readLine()) != null)
			{
				String[] splits = line.split(",");
				StringBuilder sb = new StringBuilder();
				for(int i = 0; i < splits.length -1 ; i++)
				{
					if(i == 0)
					{
						sb.append(splits[i]);
					}
					else 
					{
						sb.append(",");
						sb.append(splits[i]);
					}
				}
				String key = sb.toString();
				int count = Integer.valueOf(splits[splits.length - 1]);
				rel2NumEntities.put(key,count);
			}
			in.close();
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		
				
		try {
			in = new BufferedReader(new FileReader(typeRelCoherencePath));
			String line = null;
			while((line = in.readLine()) != null)
			{
				TypeRelCoherence trc = new TypeRelCoherence(line);
				if(trc.getCoherence() == 0)
				{
					continue;
				}
				else
				{
					trcIndex.put(trc, trc.getCoherence());
				}
				
			}
			in.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		System.out.println("Done init KBStats from reading file");
	}
	
	
	public static void init()
	{
		
	}
	
	
	
	
	
	public static long getNumEntities(String type)
	{
		if(!type2NumEntities.containsKey(type))
		{
			System.err.println("XUCHU: error, type not found in the type stats: " + type);
			return Integer.MAX_VALUE;
		}
		else
		{
			return type2NumEntities.get(type);
		}
	}
	public static boolean typeFoundInStats(String type)
	{
		if(!type2NumEntities.containsKey(type))
		{
			return false;
		}
		else{
			return true;
		}
	}
	public static Set<String> getEntitiesGivenType(KBReader reader,String type)
	{
		Set<String> entitySet =  null;
		try {
			entitySet = KBStatsManager.getAllEntities_Given_Type(reader, type);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return entitySet;
	}
	public static long getTotalNumEntities()
	{
		
		if(KnowledgeDatabaseConfig.KBStatsDirectoryBase.contains("simpleyago"))					
			return 2886351;
		else if(KnowledgeDatabaseConfig.KBStatsDirectoryBase.contains("yagodata"))	
			return 2886527;
		else if(KnowledgeDatabaseConfig.KBStatsDirectoryBase.contains("dbpediadata"))	
			return 2350906;
		else 
			return 0;
	}
	public static int getTotalNumTypes()
	{
		if(KnowledgeDatabaseConfig.KBStatsDirectoryBase.contains("simpleyago"))					
			return 6536;
		else if(KnowledgeDatabaseConfig.KBStatsDirectoryBase.contains("yagodata"))	
			return 373288;
		else if(KnowledgeDatabaseConfig.KBStatsDirectoryBase.contains("dbpediadata"))	
			return 351;
		else 
			return 0;
	}
	public static int getTotalNumRels()
	{
		if(KnowledgeDatabaseConfig.KBStatsDirectoryBase.contains("simpleyago"))					
			return 40;
		else if(KnowledgeDatabaseConfig.KBStatsDirectoryBase.contains("yagodata"))	
			return 99;
		else if(KnowledgeDatabaseConfig.KBStatsDirectoryBase.contains("dbpediadata"))	
			return 48292;
		else 
			return 0;
	}
	public static long getNumRelInstances(String rel)
	{
		String temRel = rel;
		if(rel.startsWith(TableSemantics.REL_REVERSED_TAG))
		{
			temRel = rel.substring(TableSemantics.REL_REVERSED_TAG.length());
		}
		if(!rel2NumEntities.containsKey(temRel))
		{
			System.err.println("XUCHU: error, rel not found in the rel stats" + rel);
			return 0;
		}
		else
		{
			return rel2NumEntities.get(temRel);
		}
	}
	public static boolean relFoundInStats(String rel)
	{
		String temRel = rel;
		if(rel.startsWith(TableSemantics.REL_REVERSED_TAG))
		{
			temRel = rel.substring(TableSemantics.REL_REVERSED_TAG.length());
		}
		if(!rel2NumEntities.containsKey(temRel))
		{
			return false;
		}
		else
		{
			return true;
		}
	}
	public static Set<String> getSubjectEntitesGivenRel(KBReader reader,String rel)
	{
		Set<StringPair> sps = reader.getSubjectObjectGivenRel(rel);
		Set<String> subjects = new HashSet<String>();
		for(StringPair sp: sps)
			subjects.add(sp.getS1());
		return subjects;
	}
	public static Set<String> getObjectGivenRel(KBReader reader,String rel)
	{
		Set<StringPair> sps  = reader.getSubjectObjectGivenRel(rel);
		Set<String> objects = new HashSet<String>();
		for(StringPair sp: sps)
			objects.add(sp.getS2());
		return objects;
	}
	public static long getTotalNumRelInstances()
	{
		
		if(KnowledgeDatabaseConfig.KBStatsDirectoryBase.contains("simpleyago"))					
			return 5406092;
		else if(KnowledgeDatabaseConfig.KBStatsDirectoryBase.contains("yagodata"))	
			return 12877271;
		else if(KnowledgeDatabaseConfig.KBStatsDirectoryBase.contains("dbpediadata"))	
			return 53303239;
		else 
			return 0;
	}
	/**
	 * Is type1 more specfic than type2
	 * @param type1
	 * @param type2
	 * @return
	 */
	public static boolean moreSpecificType(String type1, String type2)
	{
		long size1 = getNumEntities(type1);
		long size2 = getNumEntities(type2);
		
		if(size1 <= size2)
			return true;
		else
			return false;
	}
	/**
	 * Is re1 more speific than rel2
	 * @param rel1
	 * @param rel2
	 * @return
	 */
	public static boolean moreSpecificRel(String rel1, String rel2)
	{
		long size1 = getNumRelInstances(rel1);
		long size2 = getNumRelInstances(rel2);
		
		if(size1 <= size2)
			return true;
		else
			return false;
	}
	
	
	/**
	 * Get the semantic coherence between two types, this is an very expensive operation
	 * @param type1
	 * @param type2
	 * @return
	 */
	public static double getSemanticCoherenceTwoTypes(KBReader reader, String type1, String type2)
	{
		if(type1 == null ||type2 == null)
		{
			//System.err.println("Type 1 and Type2: " + type1 + " and " + type2);
		}
		Set<String> entitySet1 = getEntitiesGivenType(reader,type1);
		Set<String> entitySet2 = getEntitiesGivenType(reader,type2);
		
		Set<String> union = new HashSet<String>();
		union.addAll(entitySet1);
		union.addAll(entitySet2);
		int interSize = entitySet1.size() + entitySet2.size() - union.size();
		
		double res = ((double)interSize) / ((double)union.size());
		return res;
		
	}
	
	
	public static double getMaxCoherecoreGivenRel(String rel, boolean isDomainType, List<String> types){
		
		String relTemp = rel;
		if(rel.startsWith(TableSemantics.REL_REVERSED_TAG))
		  relTemp = rel.substring(TableSemantics.REL_REVERSED_TAG.length());
		
		double maxScore = 0;
		String maxType = "";
		if(types != null){
			for(String type: types){
				
				double curScore = 0;
				if(rel.startsWith(TableSemantics.REL_REVERSED_TAG)){
					curScore = getSemanticCoherenceTypeRel(type, relTemp, !isDomainType);
				}else{
					curScore = getSemanticCoherenceTypeRel(type, relTemp, isDomainType);
				}
				if(curScore > maxScore){
					maxScore = curScore;
					maxType = type;
				}
					
			}
		}
		
		System.err.println("The max coherence for rel: " + relTemp + " with domain " + isDomainType + " score: " + maxScore
				+ " with type : " + maxType);
		
		
		return maxScore;
	}
	
	public static double getMaxCoherence(){
		/*if(KnowledgeDatabaseConfig.KBStatsDirectoryBase.contains("simpleyago"))					
			return 0.5;
		else if(KnowledgeDatabaseConfig.KBStatsDirectoryBase.contains("yagodata"))	
			return 0.5;
		else if(KnowledgeDatabaseConfig.KBStatsDirectoryBase.contains("dbpediadata"))	
			return 0.5;
		else 
			return 0;*/
		double maxValue = -1;
		for(double mx: trcIndex.values()){
			if(mx > maxValue){
				maxValue = mx;
			}
		}
		return maxValue;
	}
	
	/**
	 * Get the semantic coherence between a relationship and a type, 
	 * the type could be the domain/range of the rel depending on the isDomainType
	 * @param type
	 * @param rel
	 * @param isDomainType
	 * @return
	 */
	public static double getSemanticCoherenceTypeRel(String type, String rel, boolean isDomainType)
	{
		if(type == null || rel == null)
			return 0.0;
		
		TypeRelCoherence tempTrc = new TypeRelCoherence(type,rel,isDomainType,0);
		if(trcIndex.containsKey(tempTrc))
		{
			return trcIndex.get(tempTrc);
		}
		else
		{
			return 0;
		}
		
	}
	
	
		
		
	
}
