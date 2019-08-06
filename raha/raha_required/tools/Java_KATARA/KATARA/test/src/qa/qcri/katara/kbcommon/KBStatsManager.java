package qa.qcri.katara.kbcommon;


import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;

import org.apache.lucene.index.CorruptIndexException;

import au.com.bytecode.opencsv.CSVWriter;
import qa.qcri.katara.common.test.TestUtil;
import qa.qcri.katara.kbcommon.*;
import qa.qcri.katara.kbcommon.pattern.simple.TypeRelCoherence;
import qa.qcri.katara.kbcommon.util.StringPair;


public class KBStatsManager {

	public static String typeStatsPath = null;
	public static String relStatsPath = null;
	public static String typeRelCoherencePath = null;
	public static String directSuperTypePath = null;
	public static String flatTypePath = null;
	public static String entityTypeTransitiveClosurePath = null;
	
	public static Set<String> flatTypes = null;

	public static void main(String[] args) throws Exception {

		long startTime = System.currentTimeMillis();
		/*if (args.length < 6) {
			throw new Exception("Usage: <KBDir> <RelPath> <TypesPath> <entityTypeTransitiveClosurePath> <OutputPath> <FlatTypesPath> [<DirectSuperTypesPath>]");
		}
		String kb = args[0];
		relStatsPath = args[1];
		typeStatsPath = args[2];
		entityTypeTransitiveClosurePath = args[3];
		typeRelCoherencePath = args[4];
		flatTypePath = args[5];
		if (args.length > 6)
			directSuperTypePath = args[6];
		

		System.err.println("Starting doing KB Stats collection\n"+
				" kb="+kb+"\n"+
				" relsfile="+relStatsPath+"\n"+
				" typesfile="+typeStatsPath+"\n"+
				" entityTypeTransitiveClosurePath="+entityTypeTransitiveClosurePath+"\n"+
				" outfile="+typeRelCoherencePath);*/
		
		
		

		int kbType = 4; 
		String kb = null;
		String kbStats = null;
		if(kbType == 1){
			kb = "/home/x4chu/Documents/KBs/simpleyago";
			kbStats = "/home/x4chu/Documents/KBs/simpleyagoStats/";
		} else if(kbType == 2){
			kb = "/home/x4chu/Documents/KBs/yagodata";
			kbStats = "/home/x4chu/Documents/KBs/yagodataStats/";
		} else if(kbType == 3){
			kb = "/home/x4chu/Documents/KBs/dbpediadata";
			kbStats = "/home/x4chu/Documents/KBs/dbpediadataStats/";
		}
		else if (kbType == 4)
		{
			kb = "/Users/xuchu/Documents/KBs/yagodata";
			kbStats = "/Users/xuchu/Documents/KBs/yagodataStats/";
		}
		KnowledgeDatabaseConfig.setDataDirectoryBase(kb);
		KnowledgeDatabaseConfig.KBStatsDirectoryBase = kbStats;
		
		if(KnowledgeDatabaseConfig.KBStatsDirectoryBase.endsWith(File.separator))
		{
			typeRelCoherencePath = KnowledgeDatabaseConfig.KBStatsDirectoryBase + "typeRelCohe.inx";
			typeStatsPath = KnowledgeDatabaseConfig.KBStatsDirectoryBase + "types.inx";
			relStatsPath = KnowledgeDatabaseConfig.KBStatsDirectoryBase + "rel.inx";
			directSuperTypePath = KnowledgeDatabaseConfig.KBStatsDirectoryBase + "directSuperType.inx";
			flatTypePath = KnowledgeDatabaseConfig.KBStatsDirectoryBase + "flatTypes.inx";
			entityTypeTransitiveClosurePath = KnowledgeDatabaseConfig.KBStatsDirectoryBase + "transitiveClosure.tsv";
		}
		else
		{
			typeRelCoherencePath = KnowledgeDatabaseConfig.KBStatsDirectoryBase + File.separator +  "typeRelCohe.inx";
			typeStatsPath = KnowledgeDatabaseConfig.KBStatsDirectoryBase + File.separator + "types.inx";
			relStatsPath = KnowledgeDatabaseConfig.KBStatsDirectoryBase + File.separator + "rel.inx";
			directSuperTypePath = KnowledgeDatabaseConfig.KBStatsDirectoryBase +  File.separator+  "directSuperType.inx";
			flatTypePath = KnowledgeDatabaseConfig.KBStatsDirectoryBase + File.separator+ "flatTypes.inx";
			entityTypeTransitiveClosurePath = KnowledgeDatabaseConfig.KBStatsDirectoryBase + File.separator+"transitiveClosure.tsv";
		}
		
		

		try {
			KBReader reader = new KBReader();	
			
			
			Set<String> set_entities= reader.getEntities_Direct("http://yago-knowledge.org/resource/wordnet_country_108544813");
			System.out.println("Num entieis: " + set_entities.size());
			System.out.println("country" + "," + "capital"
					+ "," + "language"
					+ "," + "currency"
					);
			for(String one_entity: set_entities){
				Set<String> caps = reader.getObjectEntitiesGivenRelAndSubject("http://yago-knowledge.org/resource/hasCapital", one_entity);
				Set<String> langs = reader.getObjectEntitiesGivenRelAndSubject("http://yago-knowledge.org/resource/hasOfficialLanguage", one_entity);
				Set<String> currs = reader.getObjectEntitiesGivenRelAndSubject("http://yago-knowledge.org/resource/hasCurrency", one_entity);
				if(caps.size() > 0 && langs.size() > 0 && currs.size() > 0){
					
					String s1 = one_entity;
					String s2 = caps.iterator().next();
					String s3 = langs.iterator().next();
					String s4 = currs.iterator().next();
					HashSet<String> s1_labels = reader.getPreferredLabels(s1);
					HashSet<String> s2_labels = reader.getLabels(s2);
					HashSet<String> s3_labels = reader.getLabels(s3);
					HashSet<String> s4_labels = reader.getLabels(s4);
					
					
					String s1_label = reader.getLabel(s1);
					String s2_label = reader.getLabel(s2);
					String s3_label = reader.getLabel(s3);
					String s4_label = reader.getLabel(s4);
					
					
					/*
					s1 = s1.substring("http://yago-knowledge.org/resource/".length()	);
					s2 = s2.substring("http://yago-knowledge.org/resource/".length()	);
				    s3 = s3.substring("http://yago-knowledge.org/resource/".length()	);
					s4 = s4.substring("http://yago-knowledge.org/resource/".length()	);
					s1 = s1.replace("_", " ");
					s2 = s2.replace("_", " ");
					s3 = s3.replace("_", " ");
					s4 = s4.replace("_", " ");
					*/
					System.out.println(s1 + "," + s2
							+ "," + s3
							+ "," + s4
							);
				}
			}
			/*
			Set<String> aa = reader.getTypes("http://dbpedia.org/resource/Christi_Lake", false);
			
			
			Set<String> sup = reader.listDirectSuperClasses("http://dbpedia.org/ontology/AdultActor");
			
			String actorType = "http://dbpedia.org/ontology/Actor";
			String adultActorType = "http://dbpedia.org/ontology/AdultActor";
			System.out.println(reader.isSuperClassOf(actorType,adultActorType));
			System.out.println("super class are: " + sup.toString());
			
			
			
			Set<String> entities = reader.getEntities_Direct(adultActorType);
			Set<String> entities_2 = reader.getEntities_Direct(actorType);
			for(String x: entities){
				if(!entities_2.contains(x)){
					System.err.println("ATTEndion");
				}
			}
			
			buildDirectSuperTypes(reader);
			//buildTypeStats(reader);
			//buildTypeStats_2(reader);
			
			//buildTypeRelIndex(reader,0,Integer.MAX_VALUE);
			
			long endTime = System.currentTimeMillis();
			System.out.println(endTime - startTime);
			System.out.println("FINISHED");
			*/
		}
		finally {
			System.err.println("Final memory: " + 
					Runtime.getRuntime().totalMemory()/(1024*1024) + " MB");
		}

	}
	
	
	
	private static void getTotalNumEntities(KBReader reader) throws  IOException
	{
		
		Map<String,Integer> types = new HashMap<String,Integer>();
		BufferedReader in = new BufferedReader(new FileReader(typeStatsPath));
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
			types.put(key,count);
		}
		in.close();
		System.out.println("Total number of types is " + types.keySet().size());
		long count = 0;
		Set<String> allEntities = new HashSet<String>();
		for(String type: types.keySet())
		{
			Set<String> typeEntities = reader.getEntities_Direct(type);
			allEntities.addAll(typeEntities);
			count += types.get(type);
		}
		
		System.out.println("Total number of entities is " + allEntities.size() + " types sum is " + count);
	}
	
	private static void getTotalNumRels(KBReader reader) throws  IOException
	{
		
		Map<String,Integer> rels = new HashMap<String,Integer>();
		BufferedReader in = new BufferedReader(new FileReader(relStatsPath));
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
			rels.put(key,count);
		}
		in.close();
		System.out.println("Total number of rels is " + rels.keySet().size());
		long count = 0;
		Set<StringPair> allsps = new HashSet<StringPair>();
		for(String rel: rels.keySet())
		{
			Set<StringPair> sps = reader.getSubjectObjectGivenRel(rel);
			allsps.addAll(sps);
			count += rels.get(rel);
		}
		
		System.out.println("Total number of sps is " + allsps.size() + " rels sum is " + count);
	}
	
	
	public static void buildTypeRelIndex(KBReader reader, int start, int end) throws IOException
	{
		Map<String,Integer> types = new HashMap<String,Integer>();
		Map<String,Integer> rels = new HashMap<String,Integer>();
		
		System.out.println("Building types hash");
		
		BufferedReader in = new BufferedReader(new FileReader(typeStatsPath));
		String line = null;
		int typeIndex = -1;
		while((line = in.readLine()) != null)
		{
			typeIndex++;
			if(typeIndex < start || typeIndex >= end)
			{
				continue;
			}
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
			types.put(key,count);
		}
		in.close();
		int typesSize = types.size();

		System.out.println("Building rels hash");

		in = new BufferedReader(new FileReader(relStatsPath));
		line = null;
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
			rels.put(key,count);
		}
		in.close();
		int relsSize = rels.size();

		System.out.println("loading the entities of all types into memory");

		Map<String,Set<String>> type2Entites = new HashMap<String,Set<String>>();
		int itype = 0;
		for(String type: types.keySet())
		{
			System.err.println("Given type ["+ (++itype) +"/"+typesSize+"] " + type);
			Set<String> entities = getAllEntities_Given_Type(reader,type);
			type2Entites.put(type, entities);
		}

		System.out.println("loaded the entities of all types into memory");
		
		PrintWriter out = new PrintWriter(new FileWriter(typeRelCoherencePath,true));
		int irel = 0;
		for(String rel: rels.keySet())
		{	
			System.err.println("Relation ["+ (++irel) +"/"+relsSize+"] " + rel);
			long startTime = System.currentTimeMillis();
			Set<StringPair> sps = reader.getSubjectObjectGivenRel(rel);

			itype = 0;
			for(String type: types.keySet())
			{
				System.err.println("Type ["+ (++itype) +"/"+typesSize+"] " + type);

				TypeRelCoherence tr1 = getCoherenceScore(reader,type,type2Entites.get(type), rel, sps,true);
				TypeRelCoherence tr2 = getCoherenceScore(reader,type,type2Entites.get(type), rel, sps,false);
				if(tr1.getCoherence() != 0)
					out.println(tr1.toString());
				if(tr2.getCoherence() != 0)	
					out.println(tr2.toString());
				
				out.flush();
			}
			long endTime = System.currentTimeMillis();
			out.flush();
			System.err.println("Relation Done " + rel + " with time " + (endTime - startTime));
		}
		out.close();
		
		
	}
	
	/**
	 * TAke care of types from [Start,end)
	 * @param reader
	 * @param start
	 * @param end
	 * @throws IOException
	 */
/*	private static void buildTypeRelIndexParallel(final KBReader reader, int start, int end) throws IOException
	{
		System.out.println("buildTypeRelIndexParallel");
		final Map<String,Integer> types = new HashMap<String,Integer>();
		final Map<String,Integer> rels = new HashMap<String,Integer>();
		
		
		BufferedReader in = new BufferedReader(new FileReader(typeStatsPath));
		String line = null;
		int typeIndex = -1;
		while((line = in.readLine()) != null)
		{
			typeIndex++;
			if(typeIndex < start || typeIndex >= end)
			{
				continue;
			}
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
			types.put(key,count);
		}
		in.close();
		
		in = new BufferedReader(new FileReader(relStatsPath));
		line = null;
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
			rels.put(key,count);
		}
		in.close();
		
		
		final Map<String,Set<String>> type2Entites = new HashMap<String,Set<String>>();
		for(String type: types.keySet())
		{
			Set<String> entities = reader.getEntities(type);
			type2Entites.put(type, entities);
		}
		System.out.println("loaded all types into memory");
		
		final Map<String,Set<StringPair>> rel2Entites = new HashMap<String,Set<StringPair>>();
		for(String rel: rels.keySet())
		{
			Set<StringPair> sps = reader.getSubjectObjectGivenRel(rel);
			rel2Entites.put(rel, sps);
		}
		System.out.println("loaded all rel into memory");
		
		
		final Map<Integer,Set<String>> thread2Rels = new HashMap<Integer,Set<String>>();
		int numThreads = 12;
		for(int i = 0; i < numThreads; i++)
		{
			thread2Rels.put(i, new HashSet<String>());
		}
		//sort the rels by their number of support
		ArrayList<String> sortedRels = new ArrayList<String>();
		for(String rel: rels.keySet())
			sortedRels.add(rel);
		Collections.sort(sortedRels, new Comparator<String>(){

			@Override
			public int compare(String o1, String o2) {
				// TODO Auto-generated method stub
				if(rels.get(o1) > rels.get(o2))
					return 1;
				else if(rels.get(o1) < rels.get(o2))
					return -1;
				else 
					return 0;
			}
			
		});
		int threadIndex = 0;
		for(int i = 0; i < sortedRels.size(); i++)
		{
			thread2Rels.get(threadIndex).add(sortedRels.get(i));
		}
		
		
		PrintWriter out = new PrintWriter(new FileWriter(typeRelCoherencePath));
		
		//parallel programming
		Map<Integer,Thread> threads = new HashMap<Integer,Thread>();
		final Map<String,Set<TypeRelCoherence>> result = new HashMap<String,Set<TypeRelCoherence>>();
		for(String rel: rels.keySet())
		{
			Set<TypeRelCoherence> temp = new HashSet<TypeRelCoherence>();
			result.put(rel,temp);
		}
		
		for(int i = 0; i < numThreads; i++)
		{
			final int tempI = i;
			Thread thread = new Thread(new Runnable()
			{
				public void run()
				{
					for(String rel: thread2Rels.get(tempI))
					{
						Set<StringPair> sps = rel2Entites.get(rel);
						for(final String type: types.keySet())
						{
							//System.out.println( rel + " AND " + type);
							TypeRelCoherence tr1 = getCoherenceScore(reader,type,type2Entites.get(type), rel, sps,true);
							TypeRelCoherence tr2 = getCoherenceScore(reader,type,type2Entites.get(type), rel, sps,false);
							result.get(rel).add(tr1);
							result.get(rel).add(tr2);
						}
					}
					
				}
			
			});
			thread.start();
			threads.put(i, thread);
			
		}
		System.out.println("Started all threads");
		for(int i = 0; i < numThreads; i++)
		{
			try {
				threads.get(i).join();
				System.out.println("Done for thread" + i);
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		for(String rel: rels.keySet())
		{
			for(TypeRelCoherence o: result.get(rel))
			{
				if(o.getCoherence() != 0)
					out.println(o);
			}
		}
		out.close();
		
	}*/
	/*public static void buildAllSubTypes(KBReader reader) throws IOException
	{
		Set<String> allTypes = reader.getAllTypes();
		System.err.println("The total number of types: " + allTypes.size());
		
		Set<String> flatTypes = new HashSet<String>();
		BufferedReader in = new BufferedReader(new FileReader(flatTypePath));
		String line = null;
		while((line = in.readLine()) != null) 
				flatTypes.add(line);
		in.close();
		System.err.println("Done loading flat types");
		
		Map<String,Set<String>> directSuperTypes = new HashMap<String,Set<String>>();
		in = new BufferedReader(new FileReader(directSuperTypePath));
		line = null;
		while((line = in.readLine()) != null) 
		{
			String[] splits = line.split("\t");
			String type = splits[0];
			Set<String> superTypes = new HashSet<String>();
			for(int i = 1; i < splits.length; i++){
				superTypes.add(splits[i]);
			}
			directSuperTypes.put(type, superTypes);
		}
		in.close();
		System.err.println("Done loading direct super types");
		
		
		
		
		String filePath = allSubTypesPath;
		try {
			int i = 0;
			PrintWriter out = new PrintWriter(new FileWriter(filePath));
			for(String type: allTypes)
			{
				i++;
				
				if(flatTypes.contains(type))
				{
					System.out.println("doing flat" + i + " : " + type);
					out.println(type + "\t");
				}
				else
				{
					System.err.println("doing non-flat" + i + " : " + type);
					 Set<String> allSubTypes = getAllSubTypes(directSuperTypes,type);
					 StringBuilder sb = new StringBuilder();
					 sb.append(type + "\t");
					 for(String temp: allSubTypes){
						 sb.append(temp + "\t");
					 }
					 out.println(sb);
				}
				out.flush();
			}
			out.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		
	}*/
	
	/*public static Set<String> getAllSubTypes(Map<String,Set<String>> directSuperTypes, String type)
	{
		Set<String> allSubTypes = new HashSet<String>();

		Set<String>  allSubTypes_Previous = null;
		
		Set<String> newOnes = new HashSet<String>();
		newOnes.add(type);
		
		do{
			
			allSubTypes_Previous = new HashSet<String>();
			allSubTypes_Previous.addAll(allSubTypes);
			
			for(String temp: directSuperTypes.keySet()){
				for(String one: newOnes){
					if(directSuperTypes.get(temp).contains(one)){
						allSubTypes.add(temp);
					}
				}
			}
			//This round has added new ones
			newOnes = new HashSet<String>();
			for(String temp: allSubTypes){
				if(!allSubTypes_Previous.contains(temp))
					newOnes.add(temp);
			}
			 
		}while(newOnes.size() > 0);
		
		
		return allSubTypes;
	}
	*/
	private static void buildDirectSuperTypes(KBReader reader)
	{
		Set<String> allTypes = reader.getAllTypes();
		System.err.println("The total number of types: " + allTypes.size());
		
		Set<String> typesWithoutAnySubclass = new HashSet<String>();
		typesWithoutAnySubclass.addAll(allTypes);
		
		String filePath = directSuperTypePath;
		try {
			int i = 0;
			PrintWriter out = new PrintWriter(new FileWriter(filePath));
			for(String type: allTypes)
			{
				if(!type.startsWith("http://"))
					continue;
				if(type.contains("\\")
						|| type.contains("#")
						|| type.contains("}")
						|| type.contains("{")
						|| type.contains(" ")
						|| type.contains("\t")
						)
						{
							continue;
						}
				i++;
				System.out.println("Doing: " + i + type);
				Set<String> directSuperTypes = reader.listDirectSuperClasses(type);
				typesWithoutAnySubclass.removeAll(directSuperTypes);
				
				StringBuilder sb = new StringBuilder();
				sb.append(type + "\t");
				for(String superType: directSuperTypes){
					sb.append(superType + "\t");
				}
				out.println(sb);
				out.flush();
			}
			out.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		System.out.println("Types without any super types: " + typesWithoutAnySubclass.size());
		try {
			int i = 0;
			PrintWriter out = new PrintWriter(new FileWriter(flatTypePath));
			for(String type: typesWithoutAnySubclass)
			{
				out.println(type);
				out.flush();
			}
			out.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}
	
	public static void buildTypeStats(KBReader reader)
	{
		Set<String> allTypes = reader.getAllTypes();
		System.err.println("The total number of types: " + allTypes.size());
		
		String filePath = typeStatsPath;
		try {
			PrintWriter out = new PrintWriter(new FileWriter(filePath));
			for(String type: allTypes)
			{
				if(!KnowledgeDatabaseConfig.interestedTypes(type))
				{
					System.err.println("Skip : " + type);
					continue;
				}
				System.out.println(type);	
				if(type.contains("\\")
				|| type.contains("#")
				|| type.contains("}")
				|| type.contains("{")
				|| type.contains(" ")
				|| type.contains("\t")
				)
				{
					continue;
				}
				out.println(type + "," + getAllEntities_Given_Type(reader,type).size());
			}
			out.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	public static void buildTypeStats_2(KBReader reader) throws IOException
	{
		Map<String,Integer> type2NumEntites = new HashMap<String,Integer>();
		
		BufferedReader in = new BufferedReader(new FileReader(entityTypeTransitiveClosurePath));
		String line = null;
		
		while((line = in.readLine())!=null){
			String[] splits = line.split("\t");
			if(splits.length != 4){
				System.err.println("ERROR");
			}
			
			String entity = splits[1];
			String entity_type = "http://yago-knowledge.org/resource/" + splits[3].replace("<","").replace(">", "");
			if(!type2NumEntites.containsKey(entity_type)){
				type2NumEntites.put(entity_type, 0);
			}else{
				type2NumEntites.put(entity_type, type2NumEntites.get(entity_type) + 1);
			}
		}
		in.close();
		System.err.println("Done reading transitive closure file. total types: " + type2NumEntites.keySet().size());
		
		PrintWriter out = new PrintWriter(new FileWriter(typeStatsPath));
		int count = 0;
		for(String type: type2NumEntites.keySet())
		{
			if(!KnowledgeDatabaseConfig.interestedTypes(type))
			{
				System.err.println("Skip : " + type);
				continue;
			}
			//System.out.println(type);	
			if(type.contains("\\")
			|| type.contains("#")
			|| type.contains("}")
			|| type.contains("{")
			|| type.contains(" ")
			|| type.contains("\t")
			)
			{
				continue;
			}
			count++;
			out.println(type + "," + type2NumEntites.get(type));
			
		}
		System.err.println("Total number of lines in types.inx: " + count);
		out.close();
		
	}
	
	public static void buildRelStats(KBReader reader)
	{
		Set<String> allRels = reader.getAllRelationships_2();
		System.out.println("Total number of rels : " + allRels.size());
		for(String rel: allRels)
		{
			System.out.println(rel);
			
		}
		
		System.out.println("***************************");
		String filePath = relStatsPath;
		try {
			PrintWriter out = new PrintWriter(new FileWriter(filePath));
			for(String rel: allRels)
			{
				if((!KnowledgeDatabaseConfig.interestedRelationships(rel))
						|| rel.equals("http://yago-knowledge.org/resource/extractionSource")
						|| rel.equals("http://yago-knowledge.org/resource/extractionTechnique"))
				{
					System.err.println("Skip : " + rel);
					continue;
				}
					
				System.out.println(rel);
				out.println(rel + "," + reader.getSubjectObjectGivenRel(rel).size());
			}
			out.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	
	public static TypeRelCoherence getCoherenceScore(KBReader reader, String type, Set<String> entities, 
			String rel, Set<StringPair> sps, boolean isDomainType)
	{
		if(entities.size() == 0 || sps.size() == 0)
		{
			TypeRelCoherence result = new TypeRelCoherence(type,rel,isDomainType,0);
			return result;
		}
		
		//Set<String> entities = reader.getEntities(type);
		Set<String> subOrObjs = new HashSet<String>();

		
		
		if(isDomainType)
		{
			for(StringPair sp: sps)
				subOrObjs.add(sp.getS1());
			
		}
		else
		{
			for(StringPair sp: sps)
				subOrObjs.add(sp.getS2());
		}
		
		Set<String> entitySet1 = entities;
		Set<String> entitySet2 = subOrObjs;
		
		double res = 0;
		
		Set<String> largerSet = (entitySet1.size() > entitySet2.size())? entitySet1: entitySet2;
		Set<String> smallerSet = (entitySet1.size() <= entitySet2.size())? entitySet1: entitySet2;
		int interSize = 0;
		for(String s: smallerSet)
		{
			if(largerSet.contains(s))
				interSize++;
		}
		
		if(false)
		{
			int unionSize = entitySet1.size() + entitySet2.size() - interSize;
			res = ((double)interSize) / unionSize;
		}
		else
		{
			if(interSize == 0)
			{
				res = 0;
			}
			else
			{
				int totalSize = 0;
				if(KnowledgeDatabaseConfig.KBStatsDirectoryBase.contains("simpleyago"))					
					totalSize =  2886351;
				else if(KnowledgeDatabaseConfig.KBStatsDirectoryBase.contains("yagodata"))	
					totalSize = 2886527;
				else if(KnowledgeDatabaseConfig.KBStatsDirectoryBase.contains("dbpediadata"))	
					totalSize = 2350906;
				double p1 = ((double)largerSet.size()) / totalSize;
				double p2 = ((double)smallerSet.size()) / totalSize;
				double p12 = ((double)interSize) / totalSize;
				double pmi = Math.log(p12 / (p1 * p2));
				double npmi = pmi / (-Math.log(p12));
				
				res = (npmi + 1) / 2;
			}	
			
		}
		
		
		
		TypeRelCoherence result = new TypeRelCoherence(type,rel,isDomainType,res);
		return result;
	}
	
	public static Set<String> getAllEntities_Given_Type(KBReader reader, String type) throws IOException
	{	
		
		if(flatTypePath == null){
			if(KnowledgeDatabaseConfig.KBStatsDirectoryBase.endsWith(File.separator)){
				flatTypePath = KnowledgeDatabaseConfig.KBStatsDirectoryBase + "flatTypes.inx";
			}
			else{	
				flatTypePath = KnowledgeDatabaseConfig.KBStatsDirectoryBase + File.separator+ "flatTypes.inx";
			}
		}
		if(entityTypeTransitiveClosurePath == null){
			if(KnowledgeDatabaseConfig.KBStatsDirectoryBase.endsWith(File.separator)){
				entityTypeTransitiveClosurePath = KnowledgeDatabaseConfig.KBStatsDirectoryBase + "transitiveClosure.tsv";
			}
			else{	
				entityTypeTransitiveClosurePath = KnowledgeDatabaseConfig.KBStatsDirectoryBase + File.separator+ "transitiveClosure.tsv";
			}
		}
	

		if(flatTypes == null){
			flatTypes = new HashSet<String>();
			BufferedReader in = new BufferedReader(new FileReader(flatTypePath));
			String line = null;
			while((line = in.readLine()) != null) 
					flatTypes.add(line);
			in.close();

		}
		
		if(flatTypes.contains(type)){
			return reader.getEntities_Direct(type);
		}
		
		
		Set<String> result = new HashSet<String>();
		
		if(!type.startsWith("http://yago-knowledge.org/resource/")){
			System.err.println("wrong type");
			System.exit(-1);
		}
		
		String type_simple = type.substring("http://yago-knowledge.org/resource/".length());
		

		
		BufferedReader in = new BufferedReader(new FileReader(entityTypeTransitiveClosurePath));
		String line = null;
		
		while((line = in.readLine())!=null){
			String[] splits = line.split("\t");
			if(splits.length != 4){
				System.err.println("ERROR");
			}
			
			String entity = splits[1];
			String entity_type = splits[3];
			if(entity_type.contains(type_simple)){
				String expaned_entity = "http://yago-knowledge.org/resource/" + entity.replace("<", "").replace(">", "");
				result.add(expaned_entity);

			}
		}
		in.close();
		System.out.println("Number of entities of " + type +  " is : " + result.size() );
		
		return result;
	
	}	

}
