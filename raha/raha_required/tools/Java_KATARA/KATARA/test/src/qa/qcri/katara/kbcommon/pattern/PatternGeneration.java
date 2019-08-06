/**
 * Author: Xu Chu
 */
package qa.qcri.katara.kbcommon.pattern;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import qa.qcri.katara.dbcommon.Table;
import qa.qcri.katara.dbcommon.Tuple;
import qa.qcri.katara.kbcommon.KnowledgeDatabaseConfig;
import qa.qcri.katara.kbcommon.KBReader;


public class PatternGeneration {

	Table table;
	
	List<Pattern> finalPatterns;
	
	KBReader reader;
	
	public PatternGeneration(Table table, KBReader reader)
	{
		this.table = table;
		this.reader = reader;
		finalPatterns = new ArrayList<Pattern>();
	}
	
	
	public void generatePattern()
	{
		//Step 1: generate base patterns for the first level
		List<Pattern> basePatterns = generateBasePattern();
		System.out.println("Step 1: Done generating base Pattern");
		
		//Step 2: Check if any type 1 base pattern is subsumed by a type 2 base pattern, if so, remove 
		
		//finalPatterns.addAll(basePatterns);
		
		//Step 3: Iteratively merging
		
		List<Pattern> expand = new ArrayList<Pattern>(basePatterns);
		int level  = 1;
		while(!expand.isEmpty())
		{
			List<Pattern> newP = new ArrayList<Pattern>();
			System.out.println("Level of expansion: " + (level++));
			for(Pattern p1: expand)
			{
				boolean canExpand = false;
				for(Pattern p2: basePatterns)
				{
					if(p1.subsume(p2))
						continue;
					
					List<Pattern> tempP = mergeTwoPatterns(p1,p2);
					if(tempP.size() > 0)
						canExpand = true;
					
					boolean duplicate = false;
					for(Pattern re: tempP)
					{
						for(Pattern re1: finalPatterns)
							if(duplicate(re,re1))
								duplicate = true;
						for(Pattern re2: newP)
							if(duplicate(re,re2))
								duplicate = true;
						if(!duplicate)
							newP.add(re);
					}
				}
				if(canExpand == false)
				{
					finalPatterns.add(p1);
				}
			}
				
			//finalPatterns.addAll(newP);
			expand = newP;
		}
		System.out.println("Step 2: Done expanding patterns");
		
	}
	
	/*
	public void generatePattern()
	{
		//Step 1: generate base patterns for the first level
		List<Pattern> basePatterns = generateBasePattern();
		
		
		//Step 2: Generate all essential patterns
		List<Pattern> level = null;
		int k = 2;
		do{
			
			//Generate candidate of the next level
			List<Pattern> candidatesK = null;
			if(k ==2){
				candidatesK = generateCandidateSize2(basePatterns);
			}else{
				candidatesK = generateCandidateSizeK(level);
			}
			
			List<Pattern> preLevel = level;
			level = new ArrayList<Pattern>();
			//Check the support for the candidateK, and only keep those candidates with enough support to level
			for(Pattern patK: candidatesK)
			{
				if(patK.support.size() > table.getNumRows() * Config.frequentPercentage)
				{
					level.add(patK);
				}
			}
			
		}while(level.isEmpty() == false);
		
		
		//Step 3: Generate all non-subsumed patterns
		
	}
	*/
	/*
	private List<Pattern> generateCandidateSize2(List<Pattern> basePatterns)
	{
		List<Pattern> result = new ArrayList<Pattern>();
		for(int i = 0; i < basePatterns.size(); i++)
			for(int j = i + 1; j < basePatterns.size(); j++)
			{
				List<Pattern> temp = mergeTwoPatterns(basePatterns.get(i),basePatterns.get(j));
				//Add the lineage info
				for(Pattern p: temp)
				{
					p.linerage.add(basePatterns.get(i));
					p.linerage.add(basePatterns.get(j));
				}
				result.addAll(temp);
			}
		return result;
	}
	private List<Pattern> generateCandidateSizeK(List<Pattern> levelSizeK_1)
	{
		List<Pattern> result = new ArrayList<Pattern>();
		for(int i = 0; i < levelSizeK_1.size(); i++)
		{
			Set<Pattern> linerage1 = levelSizeK_1.get(i).linerage;
			for(int j = i + 1; j < levelSizeK_1.size(); j++)
			{
				Set<Pattern> linerage2 = levelSizeK_1.get(j).linerage;
				
				Set<Pattern> newLinerage = new HashSet<Pattern>(linerage1);
				newLinerage.addAll(linerage2);
				
				//They share at least n-1 patterns
				if(newLinerage.size() == linerage1.size() + 1 && newLinerage.size() == linerage2.size() +1)
				{
					List<Pattern> temp = mergeTwoPatterns(levelSizeK_1.get(i),levelSizeK_1.get(j));
					//Add the linerage info
					for(Pattern p: temp)
					{
						p.linerage.addAll(newLinerage);
					}
					result.addAll(temp);
				}
			}
		}
		
		return result;
	}
	*/
	
	/**
	 * Merge two pattern, do the node mapping, etc..
	 * @param p1
	 * @param p2
	 * @return
	 */
	public List<Pattern> mergeTwoPatterns(Pattern p1, Pattern p2)
	{
		List<Pattern> result = new ArrayList<Pattern>();
		
		//Check if they can be merged: has overlapping end nodes
		List<String> newEndNodes = new ArrayList<String>();
		for(String endNode1: p1.endNodes)
		{
			newEndNodes.add(endNode1);
		}
		for(String endNode2: p2.endNodes)
		{
			if(!newEndNodes.contains(endNode2))
				newEndNodes.add(endNode2);
		}
		if(newEndNodes.size() == p1.endNodes.size() + p2.endNodes.size())
			return result;
		
		
		
		//Enumerate all possible mappings of the free nodes
		boolean [][] possibleMapping = new boolean[p1.freeNodes.size()][p2.freeNodes.size()];
		//Initialize mapping
		for(int i = 0; i < p1.freeNodes.size(); i++)
			for(int j = 0 ; j < p2.freeNodes.size(); j++)
			{
				//If V_i, and V_j are not of the same type, they cannot be mapped
				
				/*
				String type_i = p1.freeNode2Type.get(p1.freeNodes.get(i));
				String type_j = p2.freeNode2Type.get(p2.freeNodes.get(j));
				
				if(!type_i.equals(type_j))
					possibleMapping[i][j] = false;
				*/
				possibleMapping[i][j] = true;
				
			}
		
		//We should merge as much as node as possible, i.e., retain as much true as possible in the mapping
		// and at the same time ensure the mapping is valid, 	
		//i.e., one node can be mapped to at most one node
		//Check the support of the merged pattern
		for(int i = 0; i < p1.freeNodes.size(); i++)
			for(int j = 0 ; j < p2.freeNodes.size(); j++)
			{
				if(possibleMapping[i][j] == true)
				{
					boolean valid = true;
					for(Tuple tuple: table.getTuples())
					{
						//get the new assignment according to the mapping of free nodes
						if(p1.support.containsKey(tuple) && p2.support.containsKey(tuple))
						{
							boolean flag = false;
							for(Map<String,String> freeNodeAssign1: p1.support.get(tuple))
								for(Map<String,String> freeNodeAssign2: p2.support.get(tuple))
								{
									if(freeNodeAssign1.get(p1.freeNodes.get(i)).equals(freeNodeAssign2.get(p2.freeNodes.get(j))))
									{
										flag = true;
									}
								}
							if(flag == false)
							{
								valid = false;
								break;
							}
								
						}
					}
					if(valid == false)
					{
						possibleMapping[i][j] = false;
					}
				}
				
			}
		
		
		//Check the validity of the mapping, i.e., one node can be mapping to at most one other free node
		Set<Integer> usedColumns = new HashSet<Integer>();
		boolean validMapping = true;
		for(int i = 0 ; i < p1.freeNodes.size(); i++)
		{
			int count = 0;
			for(int j = 0; j < p2.freeNodes.size(); j++)
			{
				if(possibleMapping[i][j])
				{
					count++;
					if(usedColumns.contains(j))
					{
						validMapping = false;
						break;
					}
					usedColumns.add(j);
				}
					
			}
			if(count > 1)
			{
				validMapping = false;
				break;
			}
			if(!validMapping)
				break;
		}
		
		if(!validMapping)
		{
			return result;
		}
		
		//Construct the new pattern, and return the result;
		
		Pattern newPattern = mergeTwoPatternBasedOnMapping(p1,p2, possibleMapping);
		
		if(newPattern.support.keySet().size() > table.getNumRows() * KnowledgeDatabaseConfig.frequentPercentage)
			result.add(newPattern);
		
		return result;
	}

	/**
	 * Merge two pattern based on the mapping
	 * @param p1
	 * @param p2
	 * @param possibleMapping
	 * @return
	 */
	private Pattern mergeTwoPatternBasedOnMapping(Pattern p1, Pattern p2, boolean[][] possibleMapping)
	{
		
		//1. end nodes
		Pattern newPattern = new Pattern();
		newPattern.endNodes.addAll(p1.endNodes);
		for(String endNode2: p2.endNodes)
		{
			if(!newPattern.endNodes.contains(endNode2))
			{
				newPattern.endNodes.add(endNode2);
			}
		}
		
		//2.1 free nodes 
		Map<String,String> p1FreeNodes2NewFreeNodes = new HashMap<String,String>();
		Map<String,String> newFreeNodes2p1FreeNodes = new HashMap<String,String>();
		Map<String,String> p2FreeNodes2NewFreeNodes = new HashMap<String,String>();
		Map<String,String> newFreeNodes2p2FreeNodes = new HashMap<String,String>();
		
		int index = 1;
		for(int i = 0 ; i < p1.freeNodes.size(); i++)
		{
			int j = 0;
			for(j = 0; j < p2.freeNodes.size(); j++)
			{
				if(possibleMapping[i][j] == true)
				{
					break;
				}
			}
			if(j < p2.freeNodes.size())
			{
				String newFreeNode = "X" + index;
				index++;
				newPattern.freeNodes.add(newFreeNode);
				newPattern.freeNode2Type.put(newFreeNode, p1.freeNode2Type.get(p1.freeNodes.get(i)));
				
				p1FreeNodes2NewFreeNodes.put(p1.freeNodes.get(i), newFreeNode);
				p2FreeNodes2NewFreeNodes.put(p2.freeNodes.get(j), newFreeNode);
			
				newFreeNodes2p1FreeNodes.put(newFreeNode, p1.freeNodes.get(i));
				newFreeNodes2p2FreeNodes.put(newFreeNode, p2.freeNodes.get(j));
				
				
				
			}
			else
			{
				String newFreeNode = "X" + index;
				index++;
				newPattern.freeNodes.add(newFreeNode);
				newPattern.freeNode2Type.put(newFreeNode, p1.freeNode2Type.get(p1.freeNodes.get(i)));
				
				p1FreeNodes2NewFreeNodes.put(p1.freeNodes.get(i), newFreeNode);
				
				newFreeNodes2p1FreeNodes.put(newFreeNode, p1.freeNodes.get(i));
			}
		}
		for(int j = 0; j < p2.freeNodes.size(); j++)
		{
			int i = 0;
			for(i = 0; i < p1.freeNodes.size(); i++)
			{
				if(possibleMapping[i][j] == true)
				{
					break;
				}
			}
			if( i < p1.freeNodes.size())
			{
				//merged
			}
			else
			{
				String newFreeNode = "X" + index;
				index++;
				newPattern.freeNodes.add(newFreeNode);
				newPattern.freeNode2Type.put(newFreeNode, p2.freeNode2Type.get(p2.freeNodes.get(j)));
				
				p2FreeNodes2NewFreeNodes.put(p2.freeNodes.get(j), newFreeNode);
				
				newFreeNodes2p2FreeNodes.put(newFreeNode, p2.freeNodes.get(j));
			}
		}
		//2.2 get the types for free nodes
		for(String freeNode: newPattern.freeNodes)
		{
			String newType = null;
			if(newFreeNodes2p1FreeNodes.containsKey(freeNode) && newFreeNodes2p2FreeNodes.containsKey(freeNode))
			{
				String freeNode1 = newFreeNodes2p1FreeNodes.get(freeNode);
				String freeNode2 = newFreeNodes2p2FreeNodes.get(freeNode);
				if(freeNode1 == null)
					System.err.println("newFreeNodes2p1FreeNodes has error");
				if(freeNode2 == null)
					System.err.println("newFreeNodes2p2FreeNodes has error");
				String type1 = p1.freeNode2Type.get(freeNode1);
				String type2 = p2.freeNode2Type.get(freeNode2);
				if(type1 == null)
				{
					System.err.println("Merging: " + p1.toString() + " AND " + p2.toString());
					System.err.println("Pattern " + p1.toString() + " has NULl type" + newFreeNodes2p1FreeNodes.get(freeNode));
				}
				if(type2 == null)
				{
					System.err.println("Merging: " + p1.toString() + " AND " + p2.toString());
					System.err.println("Pattern " + p2.toString() + " has NULl type" + newFreeNodes2p2FreeNodes.get(freeNode));
				}
				if(reader.isSuperClassOf(type1,type2))
					newType = type2;
				else
					newType = type1;
				
			}
			else if(newFreeNodes2p1FreeNodes.containsKey(freeNode) && (!newFreeNodes2p2FreeNodes.containsKey(freeNode)))
			{
				String freeNode1 = newFreeNodes2p1FreeNodes.get(freeNode);
				if(freeNode1 == null)
					System.err.println("newFreeNodes2p1FreeNodes has error");
				String type1 = p1.freeNode2Type.get(freeNode1);
				if(type1 == null)
				{
					System.err.println("Merging: " + p1.toString() + " AND " + p2.toString());
					System.err.println("Pattern " + p1.toString() + " has NULl type" + newFreeNodes2p1FreeNodes.get(freeNode));
				}
				newType = type1;
			}
			else if((!newFreeNodes2p1FreeNodes.containsKey(freeNode)) && newFreeNodes2p2FreeNodes.containsKey(freeNode))
			{
				String freeNode2 = newFreeNodes2p2FreeNodes.get(freeNode);
				if(freeNode2 == null)
					System.err.println("newFreeNodes2p2FreeNodes has error");
				String type2 = p2.freeNode2Type.get(freeNode2);
				if(type2 == null)
				{
					System.err.println("Merging: " + p1.toString() + " AND " + p2.toString());
					System.err.println("Pattern " + p2.toString() + " has NULl type" + newFreeNodes2p2FreeNodes.get(freeNode));
				}
				newType = type2;
			}
			newPattern.freeNode2Type.put(freeNode, newType);
			
			
		}
		
		
		//3. edges
		for(Edge edge1: p1.edges)
		{
			Edge newEdge = new Edge();
			if(p1.endNodes.contains(edge1.node1))
			{
				newEdge.node1 = edge1.node1;
			}
			else
			{
				newEdge.node1 = p1FreeNodes2NewFreeNodes.get(edge1.node1);
			}
			
			if(p1.endNodes.contains(edge1.node2))
			{
				newEdge.node2 = edge1.node2;
			}
			else
			{
				newEdge.node2 = p1FreeNodes2NewFreeNodes.get(edge1.node2);
			}
			newEdge.label = edge1.label;
			if(!newPattern.edges.contains(newEdge))
			{
				newPattern.edges.add(newEdge);
			}
		}
		for(Edge edge2: p2.edges)
		{
			Edge newEdge = new Edge();
			if(p2.endNodes.contains(edge2.node1))
			{
				newEdge.node1 = edge2.node1;
			}
			else
			{
				newEdge.node1 = p2FreeNodes2NewFreeNodes.get(edge2.node1);
			}
			
			if(p2.endNodes.contains(edge2.node2))
			{
				newEdge.node2 = edge2.node2;
			}
			else
			{
				newEdge.node2 = p2FreeNodes2NewFreeNodes.get(edge2.node2);
			}
			newEdge.label = edge2.label;
			if(!newPattern.edges.contains(newEdge))
			{
				newPattern.edges.add(newEdge);
			}
		}
		
		//4. Support
		for(Tuple tuple: table.getTuples())
		{
			Set<Map<String,String>> tupleSup = new HashSet<Map<String,String> >();
			//get the new assignment according to the mapping of free nodes
			if(p1.support.containsKey(tuple) && p2.support.containsKey(tuple))
			{
				
				for(Map<String,String> freeNodeAssign1: p1.support.get(tuple))
					for(Map<String,String> freeNodeAssign2: p2.support.get(tuple))
					{
						//If for every free nodes in the new pattern, has the same assignment, then create a new assignment
						boolean validAssign = true;
						Map<String,String> newAssign = new HashMap<String,String>();
						for(String newFreeNode: newPattern.freeNodes)
						{
							if (newFreeNodes2p1FreeNodes.containsKey(newFreeNode))
							{
								String v1 = freeNodeAssign1.get(newFreeNodes2p1FreeNodes.get(newFreeNode));
								if(newFreeNodes2p2FreeNodes.containsKey(newFreeNode))
								{
									String v2 = freeNodeAssign2.get(newFreeNodes2p2FreeNodes.get(newFreeNode));
									if(!v1.equals(v2))
									{
										validAssign= false;
										break;
									}
									
								}
								newAssign.put(newFreeNode, v1);
							}
							else
							{
								assert(newFreeNodes2p2FreeNodes.containsKey(newFreeNode));
								String v2 = freeNodeAssign2.get(newFreeNodes2p2FreeNodes.get(newFreeNode));
								newAssign.put(newFreeNode, v2);
							}
							
						}
						if(validAssign)
						{
							tupleSup.add(newAssign);
						}
						
					}
				assert(tupleSup.size() > 0);
				newPattern.support.put(tuple, tupleSup);
				
					
			}
		}
		
		//5. Lineage
		newPattern.linerage.addAll(p1.linerage);
		newPattern.linerage.addAll(p2.linerage);
		
		return newPattern;
	}
	
	/**
	 * Generate base patterns
	 * @return
	 */
	public List<Pattern> generateBasePattern()
	{

		System.out.println("INFO: STARTING GENERATING  BASE PATTERN");
		
		BasePatternGeneration bp = new BasePatternGeneration(reader);
		List<Pattern> basePatternsType1 = new ArrayList<Pattern>();
		List<Pattern> basePatternsType2 = new ArrayList<Pattern>();
		
		//step 1: generate base patterns with a single end node
		for(int col1 = 0; col1 < table.getNumCols(); col1++)
		{
			basePatternsType1.addAll(bp.generateBasePattern(col1, table));
		}
		
		System.out.println("INFO: DONE GENERATING TYPE1 BASE PATTERN: " + basePatternsType1.size());
		//step 2: generate base patterns with two end nodes
		for(int col1 = 0; col1 < table.getNumCols(); col1++)
		{
			for(int col2 = col1 + 1; col2 < table.getNumCols(); col2++)
			{

				basePatternsType2.addAll(bp.generateBasePattern(col1,col2,table));
			}
		}
		System.out.println("INFO: DONE GENERATING TYPE2 BASE PATTERN: "+basePatternsType2.size());
		
		//Step 3: check if a type 1 pattern is subsumed by a type 2 pattern
		List<Pattern> basePatterns = new ArrayList<Pattern>();
		for(Pattern p1: basePatternsType1)
		{
			boolean subsumed = false;
			for(Pattern p2: basePatternsType2)
				if(p2.subsume(p1))
				{
					subsumed = true;
					break;
				}
			if(!subsumed)
				basePatterns.add(p1);
		}
		basePatterns.addAll(basePatternsType2);
		
		System.out.println("INFO: DONE GENERATING  BASE PATTERN: " + basePatterns.size());
		
		//Type checking
		for(Pattern temp: basePatterns)
		{
			for(String freeNode: temp.freeNodes)
			{
				String type = temp.freeNode2Type.get(freeNode);
				if(type == null)
				{
					System.err.println(temp.toString() + " has NULL typed free node" + freeNode);
				}
			}
		}
		System.out.println("INFO: DONE CHECKING TYPING  BASE PATTERN: " + basePatterns.size());
		
		//Step 4: Check the support
		List<Pattern> result = new ArrayList<Pattern>();
		for(Pattern p: basePatterns)
		{
			if(p.support.keySet().size() >= table.getNumRows() * KnowledgeDatabaseConfig.frequentPercentage)
				result.add(p);
		}
		System.out.println("INFO: BASE PATTERN AFTER SUPPORT: " + result.size());
		//new PatternVisualization(result);
		return result;
	}
	
	
	public List<Pattern> getFinalPatterns()
	{
		return finalPatterns;
	}
	
	/**
	 * Test if two patterns are duplicate or not by homomorphism test
	 * @param p1
	 * @param p2
	 * @return
	 */
	private boolean duplicate(Pattern p1, Pattern p2)
	{
		if(p1.subsume(p2) && p2.subsume(p1))
			return true;
		return false;
	}
	
	/**
	 * @deprecated
	 * @param p
	 */
	public void addType2FreeNode(Pattern p)
	{
		Map<Tuple, Set<Map<String, String>>> supportType = new HashMap<Tuple, Set<Map<String, String>>>();
		
		Map<String,Set<String>> freeNode2CanTypes = new HashMap<String,Set<String>>();
		for(int i = 0; i < p.freeNodes.size();i++)
			freeNode2CanTypes.put(p.freeNodes.get(i), new HashSet<String>());
		
		for(Tuple tuple: p.support.keySet())
		{
			Set<Map<String, String>> allAssType = new HashSet<Map<String,String>>();
			for(Map<String,String> oneAss: p.support.get(tuple))
			{
				Map<String,String> oneAssType = new HashMap<String,String>();
				for(String freeNode: oneAss.keySet())
				{
					String type = reader.getType(oneAss.get(freeNode));
					oneAssType.put(freeNode, type);
					freeNode2CanTypes.get(freeNode).add(type);
				}
				allAssType.add(oneAssType);
			}
			supportType.put(tuple, allAssType);
		}
		
		for(int i = 0; i < p.freeNodes.size(); i++)
		{
			String type = null;
			//Get the type!!
			p.freeNode2Type.put(p.freeNodes.get(i), type);
		}
	
	}
	
}
