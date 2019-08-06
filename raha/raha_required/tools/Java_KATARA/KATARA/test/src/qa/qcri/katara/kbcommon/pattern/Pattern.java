/**
 * Author: Xu Chu
 */
package qa.qcri.katara.kbcommon.pattern;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import qa.qcri.katara.dbcommon.Tuple;
import qa.qcri.katara.kbcommon.KnowledgeDatabaseConfig;
import qa.qcri.katara.kbcommon.KBReader;

public class Pattern implements Serializable{

	public List<String> endNodes = new ArrayList<String>();	//nodes representing values from columns, use column names
	public List<String> freeNodes = new ArrayList<String>();	//nodes representing existential variable, use x1,...xn
	public List<Edge> edges = new ArrayList<Edge>();
	
	public Map<String,String>	freeNode2Type = new HashMap<String,String>(); //Type of the free nodes
	
	
	
	public Map<Tuple, Set<Map<String, String>>> support = new HashMap<Tuple, Set<Map<String, String>>>();

	
	public Set<Pattern> linerage = new HashSet<Pattern>(); // joining of multiple patterns

	
	/**
	 * sub-graph isomorphisim testing
	 * Check if this pattern subsumes p
	 * using Ullman's algorithm
	 * @param p
	 * @return
	 */
	public boolean subsume(Pattern p)
	{
		//1. Determine is p's end nodes are a subset of this end nodes, if not return false;
		if(!endNodes.containsAll(p.endNodes))
			return false;
		
	
		//2. Generate the mapping using Ullman's method
		List<String> pFreeNodes = p.freeNodes;
		
		Pattern g = this;
		List<String> gFreeNodes = g.freeNodes;
		
		boolean [][] mapping = new boolean[pFreeNodes.size()][gFreeNodes.size()];
		//Initialize mapping
		for(int i = 0; i < pFreeNodes.size(); i++)
			for(int j = 0 ; j < gFreeNodes.size(); j++)
			{
				//If V_i, and V_j are not of the same type, they cannot be mapped
				String type_i = p.freeNode2Type.get(pFreeNodes.get(i));
				String type_j = g.freeNode2Type.get(gFreeNodes.get(j));
				
				if(type_i == null)
				{
					System.err.println("Pattern " + p.toString() + " has NULL type ");
				}
				if(type_j == null)
				{
					System.err.println("Pattern " + g.toString() + " has NULL type ");
				}
				
				if(!type_i.equals(type_j))
					mapping[i][j] = false;
				
				//If V_i's degree is greater than V_j's , cannot be mapped
				if(p.getInDegree(pFreeNodes.get(i)) > g.getInDegree(gFreeNodes.get(j)))
					mapping[i][j] = false;
				if(p.getOutDegree(pFreeNodes.get(i)) > g.getOutDegree(gFreeNodes.get(j)))
					mapping[i][j] = false;
				
				mapping[i][j] = true;
				
			}
		Set<Integer> usedColumns = new HashSet<Integer>();
		return mappingRecurse(usedColumns,0,mapping,p);
	}
	
	/**
	 * 
	 * @param usedColumns
	 * @param curRow
	 * @param mapping
	 * @param p
	 * @param pFreeNodes
	 * @param gFreeNodes
	 * @return
	 */
	private boolean mappingRecurse(Set<Integer> usedColumns, int curRow, boolean [][] mapping,Pattern p)
	{
		if(curRow == mapping.length )
		{
			if(isSubgraph(p,mapping))
			{
				return true;
			}
			else
			{
				return false;
			}
		}
		
		/*
		boolean[][] mappingNew = new boolean[mapping.length][mapping[0].length];
		for(int i = 0; i < mapping.length; i++)
			for(int j = 0; j < mapping[0].length; j++)
				mappingNew[i][j] = mapping[i][j];
		*/		
		
		boolean[] temp = new boolean[mapping[0].length];
		for(int j = 0; j < mapping[0].length; j++)
			temp[j] = mapping[curRow][j];
		
		for(int c = 0 ; c < mapping[0].length; c++)
		{
			//Choose an unusec column && curRow can be mapping to column c
			if(!usedColumns.contains(c) && mapping[curRow][c] == true)
			{
				//for the current row, set all columns to false except column c
				
				for(int j = 0; j < mapping[0].length; j++)
					if(j == c)
						mapping[curRow][j] = true;
					else 
						mapping[curRow][j] = false;
			}
			usedColumns.add(c);
			if ( mappingRecurse(usedColumns,curRow + 1, mapping, p) )
				return true;
			else
			{
				usedColumns.remove(c);
				//restore the orginal values for this column
				for(int j = 0; j < mapping[0].length; j++)
					mapping[curRow][j] = temp[j];
			}
				
		}
		
		return false;
	}
	
	/**
	 * Given the mapping, is p a subgraph
	 * @param p
	 * @param mapping
	 * @return
	 */
	private boolean isSubgraph(Pattern p, boolean [][] mapping)
	{
		List<String> pFreeNodes = p.freeNodes;
		List<String> gFreeNodes = this.freeNodes;
		
		for(Edge pe: p.edges)
		{
			String pNode1 = pe.node1;
			String pNode2 = pe.node2;
			String pLabel = pe.label;
			int pIndex1 = pFreeNodes.indexOf(pNode1);
			int pIndex2 = pFreeNodes.indexOf(pNode2);
			
			
			
			boolean exist = false;
			for(Edge ge: this.edges)
			{
				String gNode1 = ge.node1;
				String gNode2 = ge.node2;
				String gLabel = ge.label;
				int gIndex1 = gFreeNodes.indexOf(gNode1);
				int gIndex2 = gFreeNodes.indexOf(gNode2);
				
				if(!pLabel.equals(gLabel))
					continue;
				
				if ( (pIndex1 == -1 && gIndex1 != -1) //pNode1 is an end node, gNode1 is not
					||
					 (pIndex1 != -1 && gIndex1 == -1)
				   )
						continue;
				
				if ( (pIndex2 == -1 && gIndex2 != -1) //pNode1 is an end node, gNode1 is not
						||
						 (pIndex2 != -1 && gIndex2 == -1)
					   )
							continue;
					
				if(	 pIndex1 == -1 && gIndex1 == -1  )
				{
					if(!pNode1.equals(gNode1))
						continue;
					else
					{
						if(pIndex2 == -1 && gIndex2 == -1)
						{
							if(!pNode2.equals(gNode2))
								continue;
							else
							{
								exist = true;
								break;
							}
						}
						else
						{
							if(mapping[pIndex2][gIndex2])
							{
								exist = true;
								break;
							}
							else
							{
								continue;
							}
						}
					}
				}
				else
				{
					if(pIndex2 == -1 && gIndex2 == -1)
					{
						if(!pNode2.equals(gNode2))
							continue;
						else
						{
							if(mapping[pIndex1][gIndex1])
							{
								exist = true;
								break;
							}
						}
					}
					else
					{
						if(mapping[pIndex1][gIndex1] && mapping[pIndex2][gIndex2])
						{
							exist = true;
							break;
						}
					}
				}
				
			}
			if(!exist)
				return false;
		}
		return true;
	}
	
	public int getInDegree(String nodeLabel)
	{
		int result = 0;
		for(Edge e: edges)
		{
			if(e.node2.equals(nodeLabel))
				result++;
		}
		return result;
	}
	public int getOutDegree(String nodeLabel)
	{
		int result = 0;
		for(Edge e: edges)
		{
			if(e.node1.equals(nodeLabel))
				result++;
		}
		return result;
	}
	
	
	
	
	

	public void mergeSupport(Pattern p2)
	{
		for(Tuple t : p2.support.keySet())
		{
			if(support.containsKey(t))
			{
				//Merge assignments
				support.get(t).addAll(p2.support.get(t));
			}
			else
			{
				support.put(t, p2.support.get(t));
			}
		}
	}
	
	
	/**
	 * Print this pattern
	 */
	public String toString()
	{
		String s = "{";
		for(Edge e : edges)
		{
			s += "(" + e.node1 + "," + e.label + "," + e.node2 + ")";
		}
		
		s += "}";
		s += "Support: " + support.keySet().size();
		return s;
	}
	
	/*
	public HashSet<HashMap<Tuple, String>> fillTypeFromLCA_Correct(KBReader reader) throws Exception
	{
		HashSet<HashMap<Tuple, String>> assignments = new HashSet<HashMap<Tuple,String>>();
		
		Tuple[] tupArr = new Tuple[support.size()];
		tupArr = support.keySet().toArray(tupArr);
		
		findAllTypeAssignemntCombinations(reader, new HashMap<Tuple, String>(), tupArr, 0, assignments);
		
		for(HashMap<Tuple, String> assignemnt : assignments)
		{
			
		}
		return assignments;
	}
	
	public void findAllTypeAssignemntCombinations(KBReader reader, HashMap<Tuple, String> curAssignment,
			Tuple[] tuples, int pos, HashSet<HashMap<Tuple, String>> assignments)
	{
		if(pos >= support.size())
		{
			assignments.add(curAssignment);
			return;
		}
		else
		{
			Tuple t = tuples[pos];
			
			Iterator<Map<String, String>> tupAssignemntsIter = support.get(t).iterator();
			
			
			while(tupAssignemntsIter.hasNext())
			{
				HashMap<Tuple, String> curAssignmentClone = (HashMap<Tuple, String>) curAssignment.clone();
				curAssignmentClone.put(t, tupAssignemntsIter.next().get(freeNodes.get(0)));
				findAllTypeAssignemntCombinations(reader, curAssignmentClone, tuples, pos+1, assignments);
			}
		}
	}
	
	public void fillTypeFromLCA(KBReader reader) throws Exception
	{
		System.out.println("**********" + this.toString());
		if(freeNodes.size() != 1)
		{
			throw new Exception("Not a type-1 pattern");
		}
		
		
		HashSet<String> types = new HashSet<String>();
		
		for(Tuple t : support.keySet())
		{
			System.out.println("Tuple : " + t.toString() + "# assignments: " + support.get(t).size());
			for(Map<String, String> assignment : support.get(t))
			{
				for(String freeNode : assignment.keySet())
				{
					String entityURI = assignment.get(freeNode);
					System.out.println("--------: " + entityURI + " : " + reader.getType(entityURI));
					if(reader.getType(entityURI) != null)
					{
						types.add(reader.getType(entityURI));
					}
				}
			}
		}
		
		freeNode2Type.put(freeNodes.get(0), reader.getLeastCommonAncestor(types).iterator().next());
	}
	*/
	
	
	
	public void fillTypeForType1BasePattern_Xu(KBReader reader) throws Exception
	{
		System.out.println("**********fillTypeForType1BasePattern_Xu" + this.toString());
		if(freeNodes.size() != 1)
		{
			throw new Exception("Not a type-1 pattern");
		}
		
		
		HashSet<String> canTypes = new HashSet<String>();
		//Step 1: gathering candidate types
		for(Tuple t : support.keySet())
		{
			//System.out.println("Tuple : " + t.toString() + "# assignments: " + support.get(t).size());
			for(Map<String, String> assignment : support.get(t))
			{
				for(String freeNode : assignment.keySet())
				{
					String entityURI = assignment.get(freeNode);
					if(reader.getType(entityURI) != null)
					{
						canTypes.add(reader.getType(entityURI));
					}
				}
			}
		}
		//Step 2: counting the support of candidate types
		Map<String,Integer> canTypes2Sup = new HashMap<String,Integer>();
		for(String canType: canTypes)
			canTypes2Sup.put(canType, 0);
		for(Tuple t: support.keySet())
		{
			for(String canType: canTypes)
			{
				for(Map<String, String> assignment : support.get(t))
				{
					String freeNode = assignment.keySet().iterator().next(); //only one freenode
					String entityURI = assignment.get(freeNode);
					String entityType = reader.getType(entityURI);
					if(reader.isSuperClassOf(canType, entityType))
					{
						canTypes2Sup.put(canType, canTypes2Sup.get(canType) + 1);
						break;
					}
					
				}
				
			}
			
		}
		//Step 3: get the canType with the maximum support
		int max = 0;
		String finalType = KnowledgeDatabaseConfig.ROOT_CLASS;
		for(String type: canTypes)
		{
			if(canTypes2Sup.get(type) > max)
			{
				finalType = type;
				max = canTypes2Sup.get(type);
			}
		}
		
		this.freeNode2Type.put(freeNodes.get(0), finalType);
		System.out.println("----: finaltype" + finalType);
		//Step 4: Modify the assignment according to finalType
		Map<Tuple, Set<Map<String, String>>> newSupport = new HashMap<Tuple, Set<Map<String, String>>>();
		for(Tuple t: support.keySet())
		{
			for(Map<String, String> assignment : support.get(t))
			{
				String freeNode = assignment.keySet().iterator().next(); //only one freenode
				String entityURI = assignment.get(freeNode);
				String entityType = reader.getType(entityURI);
				if(entityType == null)
				{
					System.err.println("XXXXXXXXXXXXX");
				}
				if(reader.isSuperClassOf(finalType, entityType))
				{
					if(newSupport.containsKey(t))
					{
						newSupport.get(t).add(assignment);
					}
					else
					{
						Set<Map<String, String>> temp = new HashSet<Map<String,String>>();
						temp.add(assignment);
						newSupport.put(t, temp);
					}
				}
				
			}
		}
		support = newSupport;
		System.out.println("----: finaltype has new support" + support.size());
	}
	
	
	/**
	 * Fills the type of the free nodes based on domain/range information in the KB
	 * 
	 * <b>Should use for type 2 only?</b>
	 */
	public void fillTypesFromDomainAndRange(KBReader reader) throws Exception
	{
		if(endNodes.size() <= 1)
		{
			throw new Exception("Only type-2 patterns are allowed!");
		}
		
		if(freeNodes.size() <= 1)
		{
			String fnode = freeNodes.get(0);
			
			String domain1URI = reader.getDomain(edges.get(0).label);
			String domain2URI = reader.getDomain(edges.get(1).label);
			
			if(domain1URI != null && domain2URI != null)
			{
				freeNode2Type.put(fnode, reader.getLeastCommonAncestor(domain1URI, domain2URI).iterator().next());
			}
			else if(domain1URI != null && domain2URI == null)
			{
				freeNode2Type.put(fnode,domain1URI);
			}
			else if(domain1URI == null && domain2URI != null)
			{
				freeNode2Type.put(fnode,domain2URI);
			}
			else
			{
				freeNode2Type.put(fnode, KnowledgeDatabaseConfig.ROOT_CLASS);
			}
		}
		else
		{
			for(Edge e : edges)
			{
				if(!freeNodes.contains(e.node2))
				{
					continue;
				}
				
				String domainURI = reader.getDomain(e.label);
				String rangeURI = reader.getRange(e.label);
				if(domainURI != null)
				{
					if(freeNode2Type.containsKey(e.node1))
					{
						String currentType = freeNode2Type.get(e.node1);
						String lca = reader.getLeastCommonAncestor(currentType, domainURI).iterator().next();
						
						//take the more specific one
						if(lca.equals(domainURI))
						{
							freeNode2Type.put(e.node1, currentType);
						}
						else if(lca.equals(currentType))
						{
							freeNode2Type.put(e.node1, domainURI);
						}
						else
						{
							System.err.println("Inconsistent chain of range/domain in KB: choose one lca");
							freeNode2Type.put(e.node1, lca);
							//throw new Exception("Inconsistent chain of range/domain in KB");
						}
						
					}
					else
					{
						freeNode2Type.put(e.node1, domainURI);
					}
				}
				if(rangeURI != null)
				{
					if(freeNode2Type.containsKey(e.node2))
					{
						String currentType = freeNode2Type.get(e.node2);
						String lca = reader.getLeastCommonAncestor(currentType, rangeURI).iterator().next();
						
						//take the more specific one
						if(lca.equals(rangeURI))
						{
							freeNode2Type.put(e.node2, currentType);
						}
						else if(lca.equals(currentType))
						{
							freeNode2Type.put(e.node2, rangeURI);
						}
						else
						{
							System.err.println("Inconsistent chain of range/domain in KB: choose one lca");
							freeNode2Type.put(e.node2, lca);
							//throw new Exception("Inconsistent chain of range/domain in KB");
						}
						
					}
					else
					{
						freeNode2Type.put(e.node2, rangeURI);
					}
				}
				
				for(String node : freeNodes)
				{
					if(freeNode2Type.get(node) == null)
					{
						freeNode2Type.put(node, KnowledgeDatabaseConfig.ROOT_CLASS);
					}
				}
			}
		}
	}
	
	
	
	
	@Override
	public int hashCode()
	{
		final int prime = 31;
		int result = 1;
		result = prime * result + ((edges == null) ? 0 : edges.hashCode());
		result = prime * result
				+ ((endNodes == null) ? 0 : endNodes.hashCode());
		result = prime * result
				+ ((freeNodes == null) ? 0 : freeNodes.hashCode());
		return result;
	}

	
	
	@Override
	public boolean equals(Object obj)
	{
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		Pattern other = (Pattern) obj;
		if (edges == null)
		{
			if (other.edges != null)
				return false;
		}
		else if (!edges.equals(other.edges))
			return false;
		if (endNodes == null)
		{
			if (other.endNodes != null)
				return false;
		}
		else if (!endNodes.equals(other.endNodes))
			return false;
		if (freeNodes == null)
		{
			if (other.freeNodes != null)
				return false;
		}
		else if (!freeNodes.equals(other.freeNodes))
			return false;
		return true;
	}
}
