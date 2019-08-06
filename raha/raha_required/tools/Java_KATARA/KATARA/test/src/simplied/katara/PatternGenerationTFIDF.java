package simplied.katara;


import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import qa.qcri.katara.dbcommon.Table;
import qa.qcri.katara.dbcommon.Tuple;
import qa.qcri.katara.kbcommon.KBReader;
import qa.qcri.katara.kbcommon.KnowledgeDatabaseConfig;
import qa.qcri.katara.kbcommon.pattern.simple.TableSemantics;

public class PatternGenerationTFIDF {

	Table table;
	KBReader reader;
	Set<Tuple> sampleTuples = new HashSet<Tuple>();
	
	String relReversed = TableSemantics.REL_REVERSED_TAG;
	
	public PatternGenerationTFIDF(KBReader reader, Table table){
		this.table = table;
		this.reader = reader;
		if (KnowledgeDatabaseConfig.sampling >= table.getNumRows()
				|| KnowledgeDatabaseConfig.sampling == 0) {
			sampleTuples.addAll(table.getTuples());
		} else {
			for (int i = 0; i < table.getNumRows(); i++) {
				if (i + 1 > KnowledgeDatabaseConfig.sampling)
					break;
				sampleTuples.add(table.getTuple(i));
			}
		}
	}
	
	public TableSemantics getTFIDFTableSemantics() {
		TableSemantics ts = new TableSemantics();
		
		for (int col = 0; col < table.getNumCols(); col++) {
			// Get the candidate type, and supporing tuples
			String colS = String.valueOf(col);
			Map<String, Set<Tuple>> candidateTypes = getCandidateTypes(col);
			Map<String, Double> result =  setCandidateTypeScore(colS,candidateTypes) ;
			String maxT = getMaxType(result);
			ts.col2Type.put(colS, maxT);
			ts.col2TypeScore.put(colS, result.get(maxT));
		}
		
		for (int col1 = 0; col1 < table.getNumCols(); col1++) {
			for (int col2 = col1 + 1; col2 < table.getNumCols(); col2++) {
				String colFinal = col1 + "," + col2;
				Map<String, Set<Tuple>> candidateRels = getCandidateRels(col1,col2);
				Map<String, Double> result = setCandidateRelScore(colFinal,candidateRels);
				String maxT = getMaxRel(result);
				ts.col2Rel.put(colFinal, maxT);
				ts.col2RelScore.put(colFinal, result.get(maxT));
			}
		}
		return ts;

	}
	public ArrayList<TableSemantics> getTFIDFTableSemantics(int topK) {
		ArrayList<TableSemantics> topk_ts = new ArrayList<TableSemantics>();
		for(int i = 0; i < topK; i++){
			TableSemantics ts = new TableSemantics();
			topk_ts.add(ts);
		}
		
		for (int col = 0; col < table.getNumCols(); col++) {
			// Get the candidate type, and supporing tuples
			String colS = String.valueOf(col);
			Map<String, Set<Tuple>> candidateTypes = getCandidateTypes(col);
			final Map<String, Double> result =  setCandidateTypeScore(colS,candidateTypes) ;
			
			ArrayList<String> ranked = new ArrayList<String>(result.keySet());
			Collections.sort(ranked,new Comparator<String>(){

				@Override
				public int compare(String o1, String o2) {
					// TODO Auto-generated method stub
					if(result.get(o1) < result.get(o2)){
						return 1;
					}else if(result.get(o1) > result.get(o2))
						return -1;
					else{
						if(RankJoinKBStats.getNumEntities(o1) > RankJoinKBStats.getNumEntities(o2))
							return 1;
						else
							return -1;
					}
				}
				
			});
			
			for(int i = 0 ; i < topK; i++){
				TableSemantics ts = topk_ts.get(i);
				if(i < ranked.size()){
					ts.col2Type.put(colS, ranked.get(i));
					ts.col2TypeScore.put(colS, result.get(ranked.get(i)));
				}else{
					ts.col2Type.put(colS, null);
					ts.col2TypeScore.put(colS, (double) 0);
				}
				
			}
		
		}
		
		for (int col1 = 0; col1 < table.getNumCols(); col1++) {
			for (int col2 = col1 + 1; col2 < table.getNumCols(); col2++) {
				String colFinal = col1 + "," + col2;
				Map<String, Set<Tuple>> candidateRels = getCandidateRels(col1,col2);
				final Map<String, Double> result = setCandidateRelScore(colFinal,candidateRels);
				
				ArrayList<String> ranked = new ArrayList<String>(result.keySet());
				Collections.sort(ranked,new Comparator<String>(){

					@Override
					public int compare(String o1, String o2) {
						// TODO Auto-generated method stub
						if(result.get(o1) < result.get(o2)){
							return 1;
						}else if(result.get(o1) > result.get(o2))
							return -1;
						else{
							if(RankJoinKBStats.getNumRelInstances(o1) > RankJoinKBStats.getNumRelInstances(o2))
								return 1;
							else
								return -1;
						}
					}
					
				});
				
				for(int i = 0 ; i < topK; i++){
					TableSemantics ts = topk_ts.get(i);
					if(i < ranked.size()){
						ts.col2Rel.put(colFinal, ranked.get(i));
						ts.col2RelScore.put(colFinal, result.get(ranked.get(i)));
					}else{
						ts.col2Rel.put(colFinal, null);
						ts.col2RelScore.put(colFinal, (double) 0);
					}
					
				}
			
			}
		}
		return topk_ts;

	}
	private String getMaxType(Map<String, Double> result){
		
		String maxT = null;
		double max = Double.MIN_VALUE;
		for(String key : result.keySet()){
			if(result.get(key) > max){
				maxT = key;
				max = result.get(key);
			}
			else if(result.get(key) == max){
				if(RankJoinKBStats.moreSpecificType(key,maxT)){
					maxT = key;
					max = result.get(key);
				}
			}
		}
		return maxT;
	}
	private String getMaxRel(Map<String, Double> result){
		
		String maxT = null;
		double max = Double.MIN_VALUE;
		for(String key : result.keySet()){
			if(result.get(key) > max){
				maxT = key;
				max = result.get(key);
			}
			else if(result.get(key) == max){
				if(RankJoinKBStats.moreSpecificRel(key,maxT)){
					maxT = key;
					max = result.get(key);
				}
			}
		}
		return maxT;
	}
	
	
	/**
	 * Find the candidate types of the column col, including super types
	 * reachable
	 * 
	 * @param col
	 * @return
	 */
	public Map<String, Set<Tuple>> getCandidateTypes(int col) {
		Map<String, Set<Tuple>> result = new HashMap<String, Set<Tuple>>();

		int i = 0;

		//not all tuples have types
		for (Tuple t : sampleTuples) {
			Set<String> candidateTypes = null;
			// allow for fuzzy matching
			if (KnowledgeDatabaseConfig.maxMatches == -1)
				candidateTypes = reader.getTypesOfEntitiesWithLabel(t
						.getCell(col));
			else
				candidateTypes = reader.getTypesOfEntitiesWithLabel(
						t.getCell(col).getValue(),
						KnowledgeDatabaseConfig.maxMatches);
			for (String candidateType : candidateTypes) {

				
				
				if (!KnowledgeDatabaseConfig.interestedTypes(candidateType))
					continue;
				if (!result.containsKey(candidateType)) {
					result.put(candidateType, new HashSet<Tuple>());
				}
				result.get(candidateType).add(t);
			}

			i++;

		}

		//filter the result, based on minimum support
		Map<String, Set<Tuple>> filtered_result = new HashMap<String, Set<Tuple>>();
		for(String type: result.keySet()){	
			if( (((double)result.get(type).size()) / sampleTuples.size()) < KnowledgeDatabaseConfig.frequentPercentage)
				continue;
			
			if(!RankJoinKBStats.typeFoundInStats(type))
				continue;
			
			filtered_result.put(type, result.get(type));
		}
		
		
		return filtered_result;

	}

	/**
	 * Set the score of each candidate type, considering (1) number of tuples
	 * support (2) for each supporting tuple, the extent of the support by
	 * considering type hierarchy
	 * 
	 * @param col
	 *            , example "0"
	 * @return
	 */
	public Map<String, Double>  setCandidateTypeScore(String col, Map<String, Set<Tuple>> candidateTypes) {
		Map<String, Double> result = new HashMap<String, Double>();
		
		
		Map<Tuple,Set<String>> tuple2Types = new HashMap<Tuple,Set<String>>();
		for (String type : candidateTypes.keySet()){
			for(Tuple tuple: candidateTypes.get(type)){
				if(!tuple2Types.containsKey(tuple))
					tuple2Types.put(tuple, new HashSet<String>());
				
				tuple2Types.get(tuple).add(type);
				
			}
		}
		
		long minNumEntities = Long.MAX_VALUE;
		long maxNumEntities = 0;
		for(String type: candidateTypes.keySet()){
			long cur = RankJoinKBStats.getNumEntities(type);
			if(cur < minNumEntities)
				minNumEntities = cur;
			if(cur > maxNumEntities)
				maxNumEntities = cur;
		}
		
		
		int totalNumDocs = RankJoinKBStats.getTotalNumTypes(); //candidateTypes.keySet().size();
		//using tf-idf to rank the types
		for(String type: candidateTypes.keySet()){
			
			double tfidf_total = 0;
			for(Tuple tuple: tuple2Types.keySet()){
				double tf = 0;
				if(tuple2Types.get(tuple).contains(type)){
					long numEntities = RankJoinKBStats.getNumEntities(type);
					tf = 1.0 / normalize_term_frequency (numEntities,minNumEntities,maxNumEntities);
				}
				double idf = Math.log10( ((double)totalNumDocs) / tuple2Types.get(tuple).size());
				double tfidf = tf * idf;
				tfidf_total += tfidf;
			}
			
			
			result.put(type, tfidf_total);
			//System.out.println("The tdidf of " + type + " is: "+ tfidf_total);
		}
		
			
			
		
		//normalized the result
		return normalize_result(result);

	}
	
	/**
	 * Normalize this to [1,n], where n is the number of tuples
	 * @param numEntities
	 * @param minNumEntities
	 * @param maxNumEntities
	 * @return
	 */
	private double normalize_term_frequency(long numEntities, long minNumEntities, long maxNumEntities)
	{
		
		if(minNumEntities == maxNumEntities)
			return 1.0;
		
		double ratio = ((double)(table.getNumRows() - 1)) / (maxNumEntities - minNumEntities);
		
		double result = 1.0 + ratio * (numEntities - minNumEntities);
		
		return result;
	}
	
	/**
	 * Find the direct relationship between col1 and col2
	 * 
	 * @param colID1
	 * @param colID2
	 * @return
	 */
	public Map<String, Set<Tuple>> getCandidateRels(int colID1, int colID2) {

		Map<String, Set<Tuple>> result = new HashMap<String, Set<Tuple>>();

		int i = 0;

		for (Tuple t : sampleTuples) {
			// Generate candidate relationship between two columns in one
			// direction

			Set<String> candidateRels = null;
			if (KnowledgeDatabaseConfig.maxMatches == -1) {
				// candidateRels =
				// reader.getDirectRelationShips(t.getCell(colID1) .getValue(),
				// t.getCell(colID2).getValue(), false);
				candidateRels = reader.getDirectRelationShips(
						t.getCell(colID1), t.getCell(colID2), true);
			} else
				candidateRels = reader.getDirectRelationShips(t.getCell(colID1)
						.getValue(), t.getCell(colID2).getValue(), false,
						KnowledgeDatabaseConfig.maxMatches);

			for (String candidateRel : candidateRels) {

				if (!KnowledgeDatabaseConfig
						.interestedRelationships(candidateRel))
					continue;
				if (!result.containsKey(candidateRel)) {
					result.put(candidateRel, new HashSet<Tuple>());
				}

				result.get(candidateRel).add(t);
			}
			// Generate candidate relationship between two columns in reserse
			// direction
			if (KnowledgeDatabaseConfig.maxMatches == -1)
				// candidateRels =
				// reader.getDirectRelationShips(t.getCell(colID2)
				// .getValue(), t.getCell(colID1).getValue(), false);
				candidateRels = reader.getDirectRelationShips(
						t.getCell(colID2), t.getCell(colID1), true);
			else
				candidateRels = reader.getDirectRelationShips(t.getCell(colID2)
						.getValue(), t.getCell(colID1).getValue(), false,
						KnowledgeDatabaseConfig.maxMatches);
			for (String candidateRel : candidateRels) {
				if (!KnowledgeDatabaseConfig
						.interestedRelationships(candidateRel))
					continue;
				String reversed = relReversed + candidateRel;
				if (!result.containsKey(reversed)) {
					result.put(reversed, new HashSet<Tuple>());
				}

				result.get(reversed).add(t);
			}
			i++;
			// System.out.println( (100.0 * i / tuples.size()) +
			// "% of the tuples completed..." + result.size() +
			// " types found.");

		}
	
		//filter the result, based on minimum support
		Map<String, Set<Tuple>> filtered_result = new HashMap<String, Set<Tuple>>();
		for(String rel: result.keySet()){	
			if( (((double)result.get(rel).size()) / sampleTuples.size()) < KnowledgeDatabaseConfig.frequentPercentage)
				continue;
			
			if(!RankJoinKBStats.relFoundInStats(rel))
				continue;
			
			filtered_result.put(rel, result.get(rel));
		}
		
		//System.out.println("We have fetched all candidate rels for col pair " + colID1 + " and " + colID2);
		
		return filtered_result;
		
	}
	
	/**
	 * Set the score of each candidate rel, considering (1) number of tuples
	 * supporting this rel (2) for each supporting tuple, the extent of the
	 * support by considering rel hierarchy
	 * 
	 * @param col
	 *            , example "0,1"
	 * @return
	 */
	public Map<String, Double>  setCandidateRelScore(String col,Map<String, Set<Tuple>> candidateRels) {
		Map<String, Double> result = new HashMap<String, Double>();
		
		int col1 = Integer.valueOf(col.split(",")[0]);
		int col2 = Integer.valueOf(col.split(",")[1]);
		//Map<String, Set<Tuple>> candidateRels = getCandidateRels(col1,col2);
		
		
		Map<Tuple,Set<String>> tuple2Rels = new HashMap<Tuple,Set<String>>();
		for (String rel : candidateRels.keySet()){
			for(Tuple tuple: candidateRels.get(rel)){
				if(!tuple2Rels.containsKey(tuple))
					tuple2Rels.put(tuple, new HashSet<String>());
				
				tuple2Rels.get(tuple).add(rel);
				
			}
			
		}
		
		
		long minNumEntities = Long.MAX_VALUE;
		long maxNumEntities = 0;
		for(String rel: candidateRels.keySet()){
			long cur = RankJoinKBStats.getNumRelInstances(rel);
			if(cur < minNumEntities)
				minNumEntities = cur;
			if(cur > maxNumEntities)
				maxNumEntities = cur;
		}
		
		
		int totalNumDocs = RankJoinKBStats.getTotalNumRels(); //candidateRels.keySet().size();
		//using tf-idf to rank the types
		for(String rel: candidateRels.keySet()){
			
			double tfidf_total = 0;
			for(Tuple tuple: tuple2Rels.keySet()){
				double tf = 0;
				if(tuple2Rels.get(tuple).contains(rel)){
					long numEntities = RankJoinKBStats.getNumRelInstances(rel);
					tf = 1.0 / normalize_term_frequency (numEntities,minNumEntities,maxNumEntities);
				}
				double idf = Math.log10(((double)totalNumDocs)/ tuple2Rels.get(tuple).size());
				double tfidf = tf * idf;
				tfidf_total += tfidf;
			}
			
			
			result.put(rel, tfidf_total);
			//System.out.println("The tdidf of " + rel + " is: "+ tfidf_total);
		}
		
			
		//normalized the result
		return normalize_result(result);
		
	}
	
	private Map<String, Double> normalize_result(Map<String, Double> result ){
		final Map<String, Double> normalized_result = new HashMap<String,Double>();
		
		double max = -1;
		double nomalization_constant = 0;
		for(String type: result.keySet()){
			nomalization_constant += result.get(type);
			if(result.get(type) > max){
				max = result.get(type);
			}
		}
			
		
		for(String type: result.keySet()){
			//double tmp = result.get(type) / nomalization_constant;
			double tmp = result.get(type) / max;
			normalized_result.put(type, tmp);
		}
		
		ArrayList<String> typesRanked = new ArrayList<String>(normalized_result.keySet());
		Collections.sort(typesRanked, new Comparator<String>(){

			@Override
			public int compare(String o1, String o2) {
				// TODO Auto-generated method stub
				if(normalized_result.get(o1) > normalized_result.get(o2))
					return 1;
				else if(normalized_result.get(o1) < normalized_result.get(o2))
					return -1;
				else
					return 0;
			}
			
		});	
		for(String type: typesRanked){
			//System.err.println(type + " : " + normalized_result.get(type));
		}
	
		return normalized_result;
	}
}
