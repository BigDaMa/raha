/**
 * Author: Xu Chu, John Morcos
 */
package qa.qcri.katara.kbcommon.pattern;

import java.util.HashMap;
import java.util.List;
import java.util.Set;

import qa.qcri.katara.dbcommon.Table;
import qa.qcri.katara.dbcommon.Tuple;
import qa.qcri.katara.kbcommon.KBReader;
import qa.qcri.katara.kbcommon.KnowledgeDatabaseConfig;

import com.hp.hpl.jena.rdf.model.Literal;
import com.hp.hpl.jena.rdf.model.Model;
import com.hp.hpl.jena.rdf.model.ModelFactory;

public class BasePatternGeneration {
	
	
	
	private KBReader reader;
	
	
	public BasePatternGeneration(KBReader reader)
	{
		this.reader = reader;
	}
	
	/**
	 * Generate the first type of base pattern
	 * @param col1
	 * @return
	 */
	public Set<Pattern> generateBasePattern(int col, Table table)
	{
		Model model = ModelFactory.createDefaultModel();
		HashMap<Pattern, Pattern> patterns = new HashMap<Pattern, Pattern>();
		
		List<Tuple> tuples = table.getTuples();
		
		for(Tuple t : tuples)
		{
			String value = t.getCell(col).getValue();
			
			//Find all free nodes that are related
			Literal endNode = model.createLiteral(value);
			
			//Literal endNode = reader.getMatchingNode(value);
			//ArrayList<Literal> endNodes = reader.getMatchingNodes(value, 10);
			
//			for(Literal endNode : endNodes)
//			{
			
			
				Set<Pattern> newPatterns = reader.findType1Patterns(endNode, col, t);
				
				for(Pattern p : newPatterns)
				{
					if(patterns.containsKey(p))
					{
						patterns.get(p).mergeSupport(p);
					}
					else
					{
						patterns.put(p, p);
					}
				}
//			}
		}
		
		for(Pattern p : patterns.keySet())
		{
			try
			{
				//p.fillTypeFromLCA(reader);
				p.fillTypeForType1BasePattern_Xu(reader);
			}
			catch(Exception e)
			{
				e.printStackTrace();
			}
		}
		
		
		return patterns.keySet();
	}
	
	/**
	 * Generate the second type of base patterns
	 * @param col1
	 * @param col2
	 * @return
	 */
	public Set<Pattern> generateBasePattern(int col1, int col2, Table table)
	{
		Model model = ModelFactory.createDefaultModel();
		
		HashMap<Pattern, Pattern> patterns = new HashMap<Pattern, Pattern>();
		
		List<Tuple> tuples = table.getTuples();
		
		for(Tuple t : tuples)
		{
			String val1 = t.getCell(col1).getValue();
			String val2 = t.getCell(col2).getValue();
			
			if(val1.contains("\""))
			{
				
			}
			
			Literal endNode1 = model.createLiteral(val1);
			Literal endNode2 = model.createLiteral(val2);
			
			Set<Pattern> newPatterns = reader.findType2PatternsOptimized(endNode1, col1, endNode2, col2, KnowledgeDatabaseConfig.maxLength, t);
			for(Pattern p : newPatterns)
			{
				if(patterns.containsKey(p))
				{
					patterns.get(p).mergeSupport(p);
				}
				else
				{
					patterns.put(p, p);
				}
			}
			
		}
		
		for(Pattern p : patterns.keySet())
		{
			try
			{
				p.fillTypesFromDomainAndRange(reader);
				System.out.println(p.toString());
				for(String freeNode: p.freeNodes)
				{
					if(p.freeNode2Type.containsKey(freeNode))
					{
						System.out.println(freeNode + " Type: " + p.freeNode2Type.get(freeNode));
					}
				}
			}
			catch(Exception e)
			{
				e.printStackTrace();
			}
		}
		
		return patterns.keySet();
	}
	
	@Override
	protected void finalize() throws Throwable
	{
		reader.close();
	}
}
