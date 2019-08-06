package qa.qcri.katara.kbcommon.pattern;

import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.Map;

import qa.qcri.katara.dbcommon.Tuple;
import qa.qcri.katara.kbcommon.KBReader;

import com.hp.hpl.jena.rdf.model.Model;
import com.hp.hpl.jena.rdf.model.RDFNode;
import com.hp.hpl.jena.rdf.model.Resource;


/**Represents a path from source to sink*/
public class Path {
	
	private LinkedList<RDFNode> nodes = new LinkedList<RDFNode>();
	private LinkedList<PathEdge> edges = new LinkedList<Path.PathEdge>(); 
	
	
	public Path(RDFNode firstNode)
	{
		nodes.add(firstNode);
	}
	
	public boolean extendPath(PathEdge e) throws IllegalArgumentException
	{
		RDFNode lastNode = getLastNode();
		if(e.reverse && !e.n2.equals(lastNode) || !e.reverse && !e.n1.equals(lastNode))
		{
			throw new IllegalArgumentException("Path must extend from last node...");
		}
		
		
		RDFNode newNode;
		
		edges.add(e);
		if(e.reverse)
		{
			newNode = e.n1;
		}
		else
		{
			newNode = e.n2;
		}
		
		
		//If a non-simple path
		if(nodes.contains(newNode))
		{
			return false;
		}
		else
		{
			nodes.add(newNode);
			return true;
		}
	}
	
	public boolean extendPathBackwards(PathEdge e) throws IllegalArgumentException
	{
		RDFNode firstNode = getFirstNode();
		if(e.reverse && !e.n1.equals(firstNode) || !e.reverse && !e.n2.equals(firstNode))
		{
			throw new IllegalArgumentException("Backward path must extend from first node...");
		}
		
		
		RDFNode newNode;
		
		edges.addFirst(e);
		if(e.reverse)
		{
			newNode = e.n2;
		}
		else
		{
			newNode = e.n1;
		}
		
		
		//If a non-simple path
		if(nodes.contains(newNode))
		{
			return false;
		}
		else
		{
			nodes.addFirst(newNode);
			return true;
		}
	}
	
	
	/**
	 * Combines two paths, p1 and p2. The last node in p1 must be the first in p2
	 * @param p1
	 * @param p2
	 * @return The combined path
	 */
	public static Path combinePaths(Path p1, Path p2)
	{
		Path p = p1.clone();
		
		p.nodes.removeLast(); //remove to replace with the same instance that is the first of p2 and to avoid duplicates
		PathEdge lastP1Edge = p.edges.getLast();
		if(lastP1Edge.reverse)
		{
			lastP1Edge.n1 = p2.getFirstNode();
		}
		else
		{
			lastP1Edge.n2 = p2.getFirstNode();
		}
		
		p.nodes.addAll(p2.nodes);
		p.edges.addAll(p2.edges);
		
		return p;
	}
	
	
	public RDFNode getFirstNode() {
		return nodes.get(0);
	}

	public RDFNode getLastNode() {
		return nodes.get(nodes.size() - 1);
	}

	public LinkedList<PathEdge> getEdges() {
		return edges;
	}


	@SuppressWarnings("unchecked")
	@Override
	public Path clone()
	{
		Path path2 = new Path(getFirstNode());
		path2.nodes = (LinkedList<RDFNode>) nodes.clone();
		path2.edges = (LinkedList<PathEdge>) edges.clone();
		return path2;
	}

	
	/**
	 * Builds a pattern from this path. Assumes a complete path, i.e. from an end node to another
	 * @return
	 */
	public Pattern buildPattern(int col1, int col2, Model model, Tuple t)
	{
		HashMap<String, String> assignment = new HashMap<String, String>();
		
		int freeVarCounter = 1;
		
		Pattern p = new Pattern();
		
		p.endNodes.add(t.cm.positionToName(col1));
		p.endNodes.add(t.cm.positionToName(col2));
		
		//First edge....and it's always reverse (literals can be only objects)
		assert edges.get(0).reverse;
		
		Edge e = new Edge();
		e.node2 = t.cm.positionToName(col1);
		p.freeNodes.add("X1");
		e.node1 = "X1";
		e.label = edges.get(0).p.asResource().getURI();
		p.edges.add(e);
		
		
		//Add x1 to the assignment
		assignment.put("X1", edges.get(0).n1.asResource().getURI());
		
		
		String type = null;
		
		Resource rdfClass = edges.get(0).n1.asResource()
				.getPropertyResourceValue(model.createProperty(KBReader.rdfPrefix + "type"));
		if(rdfClass != null)
		{
			type = rdfClass.getURI();
		}
//		p.freeNode2Type.put("X" + freeVarCounter, type);
		freeVarCounter++;
		
		for(int i = 1; i < edges.size() - 1; i++)
		{
			e = new Edge();
			
			String assignedValue;
			type = null;
			if(edges.get(i).reverse)
			{
				e.node1 = "X" + freeVarCounter;
				e.node2 = "X" + (freeVarCounter - 1);
				
			
				assignedValue = edges.get(i).n1.asResource().getURI();
				
				
				rdfClass = edges.get(i).n1.asResource()
						.getPropertyResourceValue(model.createProperty("rdf:type"));
				if(rdfClass != null)
				{
					type = rdfClass.getURI();
				}
				
			}
			else
			{
				e.node1 = "X" + (freeVarCounter - 1);
				e.node2 = "X" + freeVarCounter;
				
				
				assignedValue = edges.get(i).n2.asResource().getURI();
				
				
				rdfClass = edges.get(i).n2.asResource()
						.getPropertyResourceValue(model.createProperty("rdf:type"));
				if(rdfClass != null)
				{
					type = rdfClass.getURI();
				}
			}
			
//			p.freeNode2Type.put("X" + freeVarCounter, type);
			p.freeNodes.add("X" + freeVarCounter);
			e.label = edges.get(i).p.asResource().getURI();
			p.edges.add(e);
			
			//add Xi to the assignment
			assignment.put("X" + freeVarCounter, assignedValue);
			
			freeVarCounter++;
			
		}
		
		
		//Last edge, again, cannot be reverse
		e = new Edge();
		e.node2 = t.cm.positionToName(col2);
		e.node1 = "X" + (freeVarCounter-1);
		e.label = edges.get(edges.size()-1).p.asResource().getURI();
		p.edges.add(e);
		
		
		
		 
		
		//Add support information from this tuple
		if(! p.support.containsKey(t))
		{
			p.support.put(t, new HashSet<Map<String, String>>());
		}
		
		p.support.get(t).add(assignment);
		
		return p;
	}

	public class PathEdge
	{
		public boolean reverse; //is the edge directed in reverse direction?
		
		public RDFNode n1;
		public RDFNode n2;
		public RDFNode p;
		
	}
}
