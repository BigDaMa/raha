/**
 * @author john
 */

package qa.qcri.katara.kbcommon;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.Map;
import java.util.Queue;
import java.util.Set;
import java.util.StringTokenizer;

import org.apache.jena.larq.IndexLARQ;
import org.apache.jena.larq.LARQ;
import org.apache.jena.larq.assembler.AssemblerLARQ;
import org.apache.log4j.Logger;

import qa.qcri.katara.dbcommon.Cell;
import qa.qcri.katara.dbcommon.Tuple;
import qa.qcri.katara.kbcommon.pattern.Edge;
import qa.qcri.katara.kbcommon.pattern.Path;
import qa.qcri.katara.kbcommon.pattern.Path.PathEdge;
import qa.qcri.katara.kbcommon.pattern.Pattern;
import qa.qcri.katara.kbcommon.util.StringPair;

import com.hp.hpl.jena.query.Dataset;
import com.hp.hpl.jena.query.Query;
import com.hp.hpl.jena.query.QueryExecution;
import com.hp.hpl.jena.query.QueryExecutionFactory;
import com.hp.hpl.jena.query.QueryFactory;
import com.hp.hpl.jena.query.QueryParseException;
import com.hp.hpl.jena.query.QuerySolution;
import com.hp.hpl.jena.query.ReadWrite;
import com.hp.hpl.jena.query.ResultSet;
import com.hp.hpl.jena.rdf.model.Literal;
import com.hp.hpl.jena.rdf.model.Model;
import com.hp.hpl.jena.rdf.model.Resource;
import com.hp.hpl.jena.tdb.TDB;
import com.hp.hpl.jena.tdb.TDBFactory;
import com.hp.hpl.jena.util.URIref;

public class KBReader implements AutoCloseable {
	protected Dataset dataset;
	protected Model model;
	protected IndexLARQ index;

	public Dataset getDataset() {
		return dataset;
	}

	public void setDataset(Dataset dataset) {
		this.dataset = dataset;
	}

	public Model getModel() {
		return model;
	}

	public void setModel(Model model) {
		this.model = model;
	}

	public IndexLARQ getIndex() {
		return index;
	}

	public void setIndex(IndexLARQ index) {
		this.index = index;
	}

	public String getPrefixes() {
		return prefixes;
	}

	public static String getRdfprefix() {
		return rdfPrefix;
	}

	protected final String prefixes = KnowledgeDatabaseConfig.prefixes;
	protected final static Logger logger = Logger.getLogger(KBReader.class);

	public static final String rdfPrefix = "http://www.w3.org/1999/02/22-rdf-syntax-ns#";
	// public static final String rdfsPrefix =
	// "http://www.w3.org/2000/01/rdf-schema#";
	// public static final String owlPrefix = "http://www.w3.org/2002/07/owl#";

	// TODO: add single quotes as a special character?
	protected static final String LUCENE_ESCAPE_CHARS = "[\\\\+\\-\\!\\(\\)\\:\\^\\]\\{\\}\\~\\*\\?\\[\\]]";
	protected static final java.util.regex.Pattern LUCENE_PATTERN = java.util.regex.Pattern
			.compile(LUCENE_ESCAPE_CHARS);
	protected static final String REPLACEMENT_STRING = "\\\\\\\\$0";

	public static final double FUZZINESS_THRESHOLD = 0.7;
	
	
	
	
	public KBReader() throws PatternDiscoveryException {
		try {

			dataset = TDBFactory.createDataset(KnowledgeDatabaseConfig
					.getDataDirectory());

			TDB.sync(dataset);
			model = dataset.getDefaultModel();
			//model.setNsPrefix("", KnowledgeDatabaseConfig.defaultNS);
			index = AssemblerLARQ.make(dataset,
					KnowledgeDatabaseConfig.getIndexDirectory());
			LARQ.setDefaultIndex(index);
			
			logger.debug("KBReader setup completed");
		} catch (Exception exc) {
			

			throw new PatternDiscoveryException(exc.getMessage());
			
		}
	}

	/**
	 * For type-1 base-patterns
	 * 
	 * @param r
	 * @param col
	 *            column number, to name the variable
	 * @return
	 */
	public Set<Pattern> findType1Patterns(Literal endNode, int col, Tuple t) {
		// String queryString = prefixes
		// + "SELECT ?x ?p WHERE {?obj pf:textMatch (\"" +endNode.getValue()+
		// "\" 1).\n"
		// + "?x ?p ?obj}";

		HashMap<Pattern, Pattern> patterns = new HashMap<Pattern, Pattern>();
		dataset.begin(ReadWrite.READ);
		try {
			String queryString = prefixes + "SELECT ?x ?p WHERE {?x ?p \""
					+ endNode.getValue() + "\""
					+ KnowledgeDatabaseConfig.languageTag + "}";
			Query query = QueryFactory.create(queryString);
			QueryExecution qexec = QueryExecutionFactory.create(query, model);
			LARQ.setDefaultIndex(qexec.getContext(), index);

			ResultSet rs = qexec.execSelect();
			try {
				while (rs.hasNext()) {

					// Statement stmt = stmtIter.next();
					QuerySolution qs = rs.next();

					if (qs.getResource("p")
							.toString()
							.equalsIgnoreCase(
									"http://yago-knowledge.org/resource/linksTo")) {
						continue;
					}

					String type = null;

					Pattern p = new Pattern();
					p.freeNodes.add("X1");
					p.freeNode2Type.put("X1", type);
					p.endNodes.add(t.cm.positionToName(col));

					Edge e = new Edge();
					e.node1 = "X1";
					e.node2 = t.cm.positionToName(col);
					// e.label = stmt.getPredicate().getURI();
					e.label = qs.getResource("p").getURI();
					p.edges.add(e);

					HashSet<Map<String, String>> assignments = new HashSet<Map<String, String>>();
					Map<String, String> assignment = new HashMap<String, String>();
					// assignment.put("C" + col, endNode.getValue().toString());
					assignment.put("X1", qs.getResource("x").getURI());

					assignments.add(assignment);
					p.support.put(t, assignments);

					if (patterns.containsKey(p)) {
						Pattern actualPattern = patterns.get(p);
						actualPattern.mergeSupport(p);
					} else {
						patterns.put(p, p);
					}
				}
			} finally {
				qexec.close();
			}
		} finally {
			dataset.end();
		}
		return patterns.keySet();
	}

	public Set<Pattern> findType2PatternsOptimized(Literal endNode1, int col1,
			Literal endNode2, int col2, int maxLength, Tuple t) {
		HashMap<Pattern, Pattern> patterns = new HashMap<Pattern, Pattern>();

		ArrayList<Path> forwardPaths = new ArrayList<Path>();
		forwardPaths.add(new Path(endNode1));

		ArrayList<Path> backwardPaths = new ArrayList<Path>();
		backwardPaths.add(new Path(endNode2));

		int forwardLength = 0;
		int backwardLength = 0;

		while (forwardLength + backwardLength < maxLength) {
			if (forwardPaths.size() <= backwardPaths.size()) {
				forwardPaths = expandPathsForward(forwardPaths);
				forwardLength++;
			} else {
				backwardPaths = expandPathsBackwards(backwardPaths);
				backwardLength++;
			}

			// Find matches
			for (Path fpath : forwardPaths) {
				for (Path bpath : backwardPaths) {
					if (fpath.getLastNode().equals(bpath.getFirstNode())
							&& (bpath.getFirstNode().isResource() || bpath
									.getFirstNode().equals(endNode2))) {
						Path fullPath = Path.combinePaths(fpath, bpath);
						Pattern pattern = fullPath.buildPattern(col1, col2,
								model, t);

						if (patterns.containsKey(pattern)) {
							patterns.get(pattern).mergeSupport(pattern);
						} else {
							patterns.put(pattern, pattern);
						}
					}
				}
			}
		}
		return patterns.keySet();
	}

	private ArrayList<Path> expandPathsForward(ArrayList<Path> forwardPaths) {
		ArrayList<Path> resultPaths = new ArrayList<Path>();

		for (Path path : forwardPaths) {
			String val = null;
			if (path.getLastNode().isResource()) {
				val = "<"
						+ URIref.encode(path.getLastNode().asResource()
								.getURI()) + ">";
			} else if (path.getLastNode().isLiteral()) {
				val = "\"" + path.getLastNode().asLiteral().getValue() + "\""
						+ KnowledgeDatabaseConfig.languageTag;
			}
			dataset.begin(ReadWrite.READ);
			try {
				String queryString = prefixes
						+ "SELECT ?x ?p ?q ?y WHERE {{?x ?p " + val
						+ "} UNION {" + val + " ?q ?y}}";
				Query query = QueryFactory.create(queryString);
				QueryExecution qexec = QueryExecutionFactory.create(query,
						model);
				LARQ.setDefaultIndex(qexec.getContext(), index);

				ResultSet rs = qexec.execSelect();
				try {
					while (rs.hasNext()) {
						Path candidatePath = path.clone();

						QuerySolution qsol = rs.next();

						PathEdge e = path.new PathEdge();

						if (qsol.get("x") != null
								&& qsol.get("y") == null
								&& !qsol.getResource("p")
										.toString()
										.equalsIgnoreCase(
												"http://yago-knowledge.org/resource/linksTo")) {
							e.n1 = qsol.get("x");
							;
							e.n2 = path.getLastNode();
							e.p = qsol.getResource("p");
							e.reverse = true;
						} else if (qsol.get("y") != null
								&& qsol.get("x") == null
								&& !qsol.getResource("q")
										.toString()
										.equalsIgnoreCase(
												"http://yago-knowledge.org/resource/linksTo")) {
							e.n1 = path.getLastNode();
							e.n2 = qsol.get("y");
							;
							e.p = qsol.getResource("q");
							e.reverse = false;
						} else {
							continue;
						}

						if (candidatePath.extendPath(e)) {
							resultPaths.add(candidatePath);
						}
					}
				} finally {
					qexec.close();
				}
			} finally {
				dataset.end();
			}
		}

		return resultPaths;
	}

	private ArrayList<Path> expandPathsBackwards(ArrayList<Path> backwardPaths) {
		ArrayList<Path> resultPaths = new ArrayList<Path>();

		for (Path path : backwardPaths) {
			String val = null;
			if (path.getFirstNode().isResource()) {
				val = "<"
						+ URIref.encode(path.getFirstNode().asResource()
								.getURI()) + ">";
			} else if (path.getFirstNode().isLiteral()) {
				val = "\"" + path.getFirstNode().asLiteral().getValue() + "\""
						+ KnowledgeDatabaseConfig.languageTag;
			} else {
				System.err.println("Illegal node for path: " + path.toString());
				continue;
			}

			dataset.begin(ReadWrite.READ);
			try {
				String queryString = prefixes
						+ "SELECT ?x ?p ?q ?y WHERE {{?x ?p " + val
						+ "} UNION {" + val + " ?q ?y}}";
				Query query = QueryFactory.create(queryString);
				QueryExecution qexec = QueryExecutionFactory.create(query,
						model);
				LARQ.setDefaultIndex(qexec.getContext(), index);

				ResultSet rs = qexec.execSelect();
				try {
					while (rs.hasNext()) {
						Path candidatePath = path.clone();

						QuerySolution qsol = rs.next();

						PathEdge e = path.new PathEdge();

						if (qsol.get("x") != null
								&& qsol.get("y") == null
								&& !qsol.getResource("p")
										.toString()
										.equalsIgnoreCase(
												"http://yago-knowledge.org/resource/linksTo")) {
							e.n1 = qsol.get("x");
							;
							e.n2 = path.getFirstNode();
							e.p = qsol.getResource("p");
							e.reverse = false;
						} else if (qsol.get("y") != null
								&& qsol.get("x") == null
								&& !qsol.getResource("q")
										.toString()
										.equalsIgnoreCase(
												"http://yago-knowledge.org/resource/linksTo")) {
							e.n1 = path.getFirstNode();
							e.n2 = qsol.get("y");
							;
							e.p = qsol.getResource("q");
							e.reverse = true;
						} else {
							continue;
						}

						if (candidatePath.extendPathBackwards(e)) {
							resultPaths.add(candidatePath);
						}
					}
				} finally {
					qexec.close();
				}
			} finally {
				dataset.end();
			}
		}

		return resultPaths;

	}

	private Map<String, Set<String>> lcaMap = new HashMap<String, Set<String>>();

	public Set<String> getLeastCommonAncestor(String typeURI1, String typeURI2) {
		Set<String> result = new HashSet<String>();

		if (typeURI1 == null) {
			typeURI1 = KnowledgeDatabaseConfig.ROOT_CLASS;
		}
		if (typeURI2 == null) {
			typeURI2 = KnowledgeDatabaseConfig.ROOT_CLASS;
		}

		if (typeURI1.equals(KnowledgeDatabaseConfig.ROOT_CLASS)
				&& typeURI2.equals(KnowledgeDatabaseConfig.ROOT_CLASS)) {
			result.add(KnowledgeDatabaseConfig.ROOT_CLASS);
			return result;
		} else if (typeURI1.equals(KnowledgeDatabaseConfig.ROOT_CLASS)) {
			result.add(typeURI2);
			return result;
		} else if (typeURI2.equals(KnowledgeDatabaseConfig.ROOT_CLASS)) {
			result.add(typeURI1);
			return result;
		}

		if (lcaMap.containsKey(typeURI1 + "_" + typeURI2))
			return lcaMap.get(typeURI1 + "_" + typeURI2);

		Resource r1 = model.getResource(typeURI1);
		Resource r2 = model.getResource(typeURI2);

		if (r1.equals(r2)) {
			result.add(r1.getURI());
			lcaMap.put(typeURI1 + "_" + typeURI2, result);
			return result;
		}
		HashSet<Resource> r1Parents = listDirectSuperClasses(r1);
		// System.out.println(r1 + " # Parents: " + r1Parents.size());
		HashSet<Resource> r2Parents = listDirectSuperClasses(r2);

		// System.out.println(r2 + " # Parents: " + r2Parents.size());
		for (Resource r1Parent : r1Parents) {

			if (r1Parent.equals(r1))
				continue;
			Set<String> temp = getLeastCommonAncestor(r1Parent.getURI(),
					r2.getURI());
			result.addAll(temp);
		}

		for (Resource r2Parent : r2Parents) {
			if (r2Parent.equals(r2))
				continue;
			Set<String> temp = getLeastCommonAncestor(r1.getURI(),
					r2Parent.getURI());
			result.addAll(temp);
		}

		lcaMap.put(typeURI1 + "_" + typeURI2, result);

		boolean tag = true;
		while (tag) {
			tag = false;
			for (String s1 : result) {
				for (String s2 : result) {
					if (s1.equals(s2))
						continue;
					if (isSuperClassOf(s1, s2)) {
						result.remove(s1);
						tag = true;
						break;
					}
				}
				if (tag == true)
					break;

			}

		}

		return result;
	}

	public Set<String> getLeastCommonAncestor(Collection<String> typeURIs) {
		Set<String> result = new HashSet<String>();

		ArrayList<String> types = new ArrayList<String>(typeURIs);
		if (typeURIs.size() == 1) {
			result.add(types.get(0));
			return result;
		} else if (types.size() == 2) {
			result.addAll(getLeastCommonAncestor(types.get(0), types.get(1)));
			return result;
		}

		ArrayList<String> types_1 = new ArrayList<String>();
		for (int i = 1; i < types.size(); i++) {
			types_1.add(types.get(i));
		}

		Set<String> lca_1 = getLeastCommonAncestor(types_1);

		for (String temp : lca_1) {
			Set<String> tempresult = getLeastCommonAncestor(types.get(0), temp);
			result.addAll(tempresult);
		}

		boolean tag = true;
		while (tag) {
			tag = false;
			for (String s1 : result) {
				for (String s2 : result) {
					if (s1.equals(s2))
						continue;
					if (isSuperClassOf(s1, s2)) {
						result.remove(s1);
						tag = true;
						break;
					}
				}
				if (tag == true)
					break;

			}

		}

		return result;
	}

	/**
	 * @deprecated use getTypes() instead Returns the URI of the class of the
	 *             resouce of the given entity
	 * @param uri
	 * @return
	 */
	public String getType(String uri) {
		Resource type = model.getResource(uri).getPropertyResourceValue(
				model.getProperty("rdf:type"));
		if (type != null) {
			return type.getURI();
		} else {
			// return null;
			return KnowledgeDatabaseConfig.ROOT_CLASS;
		}
	}

	/**
	 * 
	 * @param uri
	 *            The entity in question
	 * @param withSuperClasses
	 *            Whether to return superclasses as well
	 * @return A set of the URIs of the classes this entity belongs to
	 */
	public HashSet<String> getTypes(String uri, boolean withSuperClasses) {
		HashSet<String> types = new HashSet<>();

		dataset.begin(ReadWrite.READ);
		try {
			String queryString = prefixes;

			if (withSuperClasses) {
				queryString += "SELECT DISTINCT ?t WHERE { <"
						+ URIref.encode(uri)
						+ "> rdf:type/rdfs:subClassOf* ?t}";
			} else {
				queryString += "SELECT DISTINCT ?t WHERE { <"
						+ URIref.encode(uri) + "> rdf:type ?t}";
			}
			Query query = QueryFactory.create(queryString);
			QueryExecution qexec = QueryExecutionFactory.create(query, model);
			LARQ.setDefaultIndex(qexec.getContext(), index);

			ResultSet rs = qexec.execSelect();
			try {
				while (rs.hasNext()) {
					types.add(rs.next().getResource("t").getURI());
				}
			} finally {
				qexec.close();
			}
		} finally {
			dataset.end();
		}

		return types;
	}

	/**
	 * @deprecated Use getLabels() instead Returns the label of the class of the
	 *             resouce of the given entity
	 * @param uri
	 * @return
	 */
	public String getLabel(String uri) {
		Resource label = model.getResource(uri).getPropertyResourceValue(
				model.getProperty("rdfs:label"));
		if (label != null) {
			return label.asLiteral().getValue().toString();
		} else {
			return null;
		}
	}
	public HashSet<String> getPreferredLabels(String uri) {

		HashSet<String> labels = new HashSet<>();

		dataset.begin(ReadWrite.READ);
		try {
			String queryString = prefixes + "SELECT DISTINCT ?l WHERE { <"
					+ URIref.encode(uri) + "> rdfs:preferredLabel ?l}";
			Query query = QueryFactory.create(queryString);
			QueryExecution qexec = QueryExecutionFactory.create(query, model);
			LARQ.setDefaultIndex(qexec.getContext(), index);

			ResultSet rs = qexec.execSelect();
			try {
				while (rs.hasNext()) {
					labels.add(rs.next().getLiteral("l").getValue().toString());
				}
			} finally {
				qexec.close();
			}
		} finally {
			dataset.end();
		}

		return labels;
	}
	/**
	 * Returns the label of the class of the resouce of the given entity
	 * 
	 * @param uri
	 * @return
	 */
	public HashSet<String> getLabels(String uri) {

		HashSet<String> labels = new HashSet<>();

		dataset.begin(ReadWrite.READ);
		try {
			String queryString = prefixes + "SELECT DISTINCT ?l WHERE { <"
					+ URIref.encode(uri) + "> rdfs:label ?l}";
			Query query = QueryFactory.create(queryString);
			QueryExecution qexec = QueryExecutionFactory.create(query, model);
			LARQ.setDefaultIndex(qexec.getContext(), index);

			ResultSet rs = qexec.execSelect();
			try {
				while (rs.hasNext()) {
					labels.add(rs.next().getLiteral("l").getValue().toString());
				}
			} finally {
				qexec.close();
			}
		} finally {
			dataset.end();
		}

		return labels;
	}

	/**
	 * Returns the label of the class of the resouce of the given entity
	 * 
	 * @param uri
	 * @return
	 */
	public HashSet<Literal> getLabelsAsLiterals(String uri) {

		HashSet<Literal> labels = new HashSet<>();

		dataset.begin(ReadWrite.READ);
		try {
			String queryString = prefixes + "SELECT DISTINCT ?l WHERE { <"
					+ uri + "> rdfs:label ?l}";
			Query query = QueryFactory.create(queryString);
			QueryExecution qexec = QueryExecutionFactory.create(query, model);
			LARQ.setDefaultIndex(qexec.getContext(), index);

			ResultSet rs = qexec.execSelect();
			try {
				while (rs.hasNext()) {
					labels.add(rs.next().getLiteral("l"));
				}
			} finally {
				qexec.close();
			}
		} finally {
			dataset.end();
		}

		return labels;
	}

	/**
	 * Returns the URI of the class that is the domain of the designated
	 * property.
	 * 
	 * @param propertyURI
	 *            URI of the property in question
	 * @return URI of the class that is the domain. Null if the property has no
	 *         defined domain.
	 */
	public String getDomain(String propertyURI) {
		Resource type = model
				.getProperty(propertyURI)
				.getPropertyResourceValue(
						model.getProperty("http://www.w3.org/2000/01/rdf-schema#domain"));
		if (type != null) {
			return type.getURI();
		} else {
			return null;
		}
	}

	/**
	 * Returns the URI of the class that is the range of the designated
	 * property.
	 * 
	 * @param propertyURI
	 *            URI of the property in question
	 * @return URI of the class that is the range. Null if the property has no
	 *         defined range.
	 */
	public String getRange(String propertyURI) {
		Resource type = model
				.getProperty(propertyURI)
				.getPropertyResourceValue(
						model.getProperty("http://www.w3.org/2000/01/rdf-schema#range"));
		if (type != null) {
			return type.getURI();
		} else {
			return null;
		}
	}

	public HashSet<Literal> getMatchingNodes(String value, double minScore) {
		HashSet<Literal> literals = new HashSet<Literal>();

		dataset.begin(ReadWrite.READ);
		try {
			String q = "PREFIX pf: <http://jena.hpl.hp.com/ARQ/property#>\n"
					+ "SELECT DISTINCT ?lit	WHERE { ?lit pf:textMatch ('"
					+ prepareForLucene(value) + "' " + minScore + ")}";

			Query query = QueryFactory.create(q);
			QueryExecution qexec = QueryExecutionFactory.create(query, model);
			LARQ.setDefaultIndex(qexec.getContext(), index);

			ResultSet rs = qexec.execSelect();

			try {
				while (rs.hasNext()) {
					literals.add(rs.next().getLiteral("lit"));
				}
			} finally {
				qexec.close();
			}
		} finally {
			dataset.end();
		}

		return literals;
	}

	// public Set<Literal> getMatchingNodes(String value, double minScore)
	// {
	// HashSet<Literal> results = new HashSet<>();
	// NodeIterator nIter = index.searchModelByIndex("('" + value + "'  + " +
	// minScore + ")");
	// for ( ; nIter.hasNext() ; )
	// {
	// // if it's an index storing literals ...
	// Literal lit = (Literal)nIter.nextNode() ;
	// results.add(lit);
	// }
	//
	// return results;
	// }

	/**
	 * Note: I make sure that the list has no duplicates
	 * 
	 * @param value
	 * @param maxMatches
	 * @return
	 */
	public ArrayList<Literal> getMatchingNodes(String value, int maxMatches) {
		ArrayList<Literal> literals = new ArrayList<Literal>();

		dataset.begin(ReadWrite.READ);
		try {
			String q = "PREFIX pf: <http://jena.hpl.hp.com/ARQ/property#>\n"
					+ "SELECT DISTINCT ?lit WHERE { ?lit pf:textMatch ('"
					+ prepareForLucene(value) + "' " + maxMatches + ")}";

			Query query = QueryFactory.create(q);
			QueryExecution qexec = QueryExecutionFactory.create(query, model);
			LARQ.setDefaultIndex(qexec.getContext(), index);

			ResultSet rs = qexec.execSelect();

			try {
				while (rs.hasNext()) {

					Literal lit = rs.next().getLiteral("lit");
					if (!literals.contains(lit)) {
						literals.add(lit);
					}
				}

			} finally {
				qexec.close();
			}
		} finally {
			dataset.end();
		}

		return literals;
	}

	// public Set<Literal> getMatchingNodes(String value, int maxMatches)
	// {
	// HashSet<Literal> results = new HashSet<>();
	// NodeIterator nIter = index.searchModelByIndex("('" + value + "'  + " +
	// maxMatches + ")");
	// for ( ; nIter.hasNext() ; )
	// {
	// // if it's an index storing literals ...
	// Literal lit = (Literal)nIter.nextNode() ;
	// results.add(lit);
	// }
	//
	// return results;
	// }

	/**
	 * Test if 1 is a superclass of 2
	 * 
	 * @param typeURI1
	 * @param typeURI2
	 * @return
	 */
	public boolean isSuperClassOf(String typeURI1, String typeURI2) {
		if (typeURI1.equals(typeURI2))
			return true;

		for (Resource sup : listDirectSuperClasses(model.getResource(typeURI2))) {
			if (sup.getURI().equals(typeURI2))
				return true;
			if (isSuperClassOf(typeURI1, sup.getURI())) {
				return true;
			}
		}

		return false;
	}
	public HashSet<String> listDirectSuperClasses(String typeurl) {
		Resource cls = model.getResource(typeurl);
		HashSet<String> superClasses = new HashSet<String>();

		String uri = cls.getURI();

		// System.err.println("HACK: added default prefix when uri does not start with http://");
		if (!uri.startsWith("http://")) {
			uri = model.getNsPrefixURI("") + uri;
		}
		uri = URIref.encode(uri);

		dataset.begin(ReadWrite.READ);
		try {
			String queryString = prefixes + "SELECT DISTINCT ?x WHERE {<" + uri
					+ "> rdfs:subClassOf ?x}";
			Query query = QueryFactory.create(queryString);
			QueryExecution qexec = QueryExecutionFactory.create(query, model);
			LARQ.setDefaultIndex(qexec.getContext(), index);

			ResultSet rs = qexec.execSelect();
			try {
				while (rs.hasNext()) {
					superClasses.add(rs.next().getResource("x").getURI());
				}
			} finally {
				qexec.close();
			}
		} finally {
			dataset.end();
		}
		return superClasses;
		// return superClasses.iterator();
	}
	public HashSet<Resource> listDirectSuperClasses(Resource cls) {
		HashSet<Resource> superClasses = new HashSet<Resource>();

		String uri = cls.getURI();

		// System.err.println("HACK: added default prefix when uri does not start with http://");
		if (!uri.startsWith("http://")) {
			uri = model.getNsPrefixURI("") + uri;
		}
		uri = URIref.encode(uri);

		dataset.begin(ReadWrite.READ);
		try {
			String queryString = prefixes + "SELECT DISTINCT ?x WHERE {<" + uri
					+ "> rdfs:subClassOf ?x}";
			Query query = QueryFactory.create(queryString);
			QueryExecution qexec = QueryExecutionFactory.create(query, model);
			LARQ.setDefaultIndex(qexec.getContext(), index);

			ResultSet rs = qexec.execSelect();
			try {
				while (rs.hasNext()) {
					superClasses.add(rs.next().getResource("x"));
				}
			} finally {
				qexec.close();
			}
		} finally {
			dataset.end();
		}
		return superClasses;
		// return superClasses.iterator();
	}

	/*******************************
	 * ******************************
	 * START IMPORTANT METHODS USING BY CORE KATARA
	 * ******************************
	 * ******************************
	 * ******************************
	 * 
	 */

	/**
	 * Check if the specifies instance (url) is of the specifies type (url)
	 * 
	 * @param instance
	 * @param type
	 * @return
	 */
	public boolean instanceChecking(String instance, String type) {

		String queryString = prefixes + "ASK " + "{ <"
				+ URIref.encode(instance) + "> rdf:type/rdfs:subClassOf* <"
				+ type + ">}";

		Query query = QueryFactory.create(queryString);
		QueryExecution qexec = QueryExecutionFactory.create(query, model);
		boolean result = qexec.execAsk();
		qexec.close();

		return result;
	}
	
	/**
	 * Check if the entity1 and entity2 are connected by a relURI
	 * @param entity1
	 * @param entity2
	 * @param relURI
	 * @return
	 */
	public boolean relInstanceChecking(String entity1, String entity2, String relURI){
		
		
		String queryString = prefixes + "ASK " + "{ <"
				+ URIref.encode(entity1) + ">" + relURI +  "<"
				+ URIref.encode(entity2) + ">}";

		Query query = QueryFactory.create(queryString);
		QueryExecution qexec = QueryExecutionFactory.create(query, model);
		boolean result = qexec.execAsk();
		qexec.close();

		return result;
	}
	public Set<String> getEntities_Direct(String typeURI) {
		Set<String> result = new HashSet<String>();

		dataset.begin(ReadWrite.READ);
		try {
			String queryString = prefixes
					+ "SELECT DISTINCT ?x WHERE { ?x rdf:type <" + typeURI
					+ ">}";
		
			
			
			Query query = QueryFactory.create(queryString);
			QueryExecution qexec = QueryExecutionFactory.create(query, model);
			LARQ.setDefaultIndex(qexec.getContext(), index);

			ResultSet rs = qexec.execSelect();
			try {
				while (rs.hasNext()) {
					result.add(rs.next().getResource("x").getURI());
				}
			} finally {
				qexec.close();
			}
		} finally {
			dataset.end();
		}

		return result;
	}
	
	
	
	
	/**
	 * Get the subject entities in the KB of the specified relationship
	 * 
	 * @param rel
	 * @return
	 */
	public Set<String> getSubjectEntities(String rel) {
		Set<String> result = new HashSet<String>();

		dataset.begin(ReadWrite.READ);
		try {
			String queryString = prefixes + "SELECT DISTINCT ?x WHERE { ?x <"
					+ URIref.encode(rel) + "> ?y}";
			Query query = QueryFactory.create(queryString);
			QueryExecution qexec = QueryExecutionFactory.create(query, model);
			LARQ.setDefaultIndex(qexec.getContext(), index);

			ResultSet rs = qexec.execSelect();
			try {
				while (rs.hasNext()) {
					result.add(rs.next().getResource("x").getURI());
				}
			} finally {
				qexec.close();
			}
		} finally {
			dataset.end();
		}

		return result;
	}
	/**
	 * Get the subject entities in the KB of the specified relationship, and object
	 * 
	 * @param rel
	 * @return
	 */
	public Set<String> getSubjectEntitiesGivenRelAndObject(String rel, String object) {
		Set<String> result = new HashSet<String>();

		dataset.begin(ReadWrite.READ);
		try {
			String queryString = prefixes + "SELECT DISTINCT ?x WHERE { ?x <"
					+ URIref.encode(rel) + ">" + " <" + URIref.encode(object) + ">" + "}";
			Query query = QueryFactory.create(queryString);
			QueryExecution qexec = QueryExecutionFactory.create(query, model);
			LARQ.setDefaultIndex(qexec.getContext(), index);

			ResultSet rs = qexec.execSelect();
			try {
				while (rs.hasNext()) {
					result.add(rs.next().getResource("x").getURI());
				}
			} finally {
				qexec.close();
			}
		} finally {
			dataset.end();
		}

		return result;
	}
	/**
	 * Get the objects entities in the KB of the specified relationship, and  subject
	 * 
	 * @param rel
	 * @return
	 */
	public Set<String> getObjectEntitiesGivenRelAndSubject(String rel, String subject) {
		Set<String> result = new HashSet<String>();

		dataset.begin(ReadWrite.READ);
		try {
			String queryString = prefixes + "SELECT DISTINCT ?x WHERE {<" + URIref.encode(subject)+  "><"
					+ URIref.encode(rel) + "> ?x}";
			Query query = QueryFactory.create(queryString);
			QueryExecution qexec = QueryExecutionFactory.create(query, model);
			LARQ.setDefaultIndex(qexec.getContext(), index);

			ResultSet rs = qexec.execSelect();
			try {
				while (rs.hasNext()) {
					result.add(rs.next().getResource("x").getURI());
				}
			} finally {
				qexec.close();
			}
		} finally {
			dataset.end();
		}

		return result;
	}

	/**
	 * Get the subject/object entities in the KB of the specified relationship
	 * 
	 * @param rel
	 * @return
	 */
	public Set<StringPair> getSubjectObjectGivenRel(String rel) {
		Set<StringPair> result = new HashSet<StringPair>();

		dataset.begin(ReadWrite.READ);
		try {
			String queryString = prefixes
					+ "SELECT DISTINCT ?x ?y WHERE { ?x <" + URIref.encode(rel) //+ "/rdfs:subPropertyOf*" 
					+ "> ?y}";
			Query query = QueryFactory.create(queryString);
			QueryExecution qexec = QueryExecutionFactory.create(query, model);
			LARQ.setDefaultIndex(qexec.getContext(), index);

			ResultSet rs = qexec.execSelect();
			try {
				while (rs.hasNext()) {
					QuerySolution qs = rs.next();
					String subject = qs.getResource("x").getURI();
					String object = null;
					if (qs.get("y").isLiteral())
						object = qs.getLiteral("y").toString();
					else if (qs.get("y").isResource())
						object = qs.getResource("y").getURI();
					result.add(new StringPair(subject, object));
				}
			} finally {
				qexec.close();
			}
		} finally {
			dataset.end();
		}

		return result;
	}
	/**
	 * 
	 * @param cell
	 * @return
	 */
	public Set<String> getTypesOfEntitiesWithLabel_DirectType(Cell cell) {
		
		Set<String> result = new HashSet<String>();

		String label = cell.getValue();
		String type = cell.getType();
		if(!type.equalsIgnoreCase("String"))
			return result;
		else{
			label = "\"" + label + "\"" + KnowledgeDatabaseConfig.languageTag ;
		}
			
		if(label.equals(""))
			return result;
		dataset.begin(ReadWrite.READ);
		try {
			String queryString = prefixes
					+ "SELECT DISTINCT ?t WHERE { ?x rdfs:label " + label + ".\n"
					+ "?x rdf:type ?t }";
			Query query = QueryFactory.create(queryString);
			QueryExecution qexec = QueryExecutionFactory.create(query, model);
			LARQ.setDefaultIndex(qexec.getContext(), index);

			ResultSet rs = qexec.execSelect();
			try {
				while (rs.hasNext()) {

					QuerySolution qs = rs.next();
					result.add(qs.getResource("t").getURI());

				}
			}finally {
				qexec.close();
			}
		}catch(QueryParseException e){
			//System.err.println("Query parse exception, ok to skip");
		}
		catch(Exception e){
			System.out.println(e.toString());
		} finally {
			dataset.end();
		}
		
		if(KnowledgeDatabaseConfig.dataDirectoryBase.toLowerCase().contains("dbpedia")){
			dataset.begin(ReadWrite.READ);
			try {
				/*String queryString = prefixes
						+ "SELECT DISTINCT ?t WHERE { ?x rdfs:label ?l" + ".\n"
						+ "?x <http://dbpedia.org/ontology/wikiPageDisambiguates> ?y.\n"
						+ "?y rdf:type/rdfs:subClassOf* ?t .\n"
						+ "FILTER( ?l = " + label + " || ?l = "  + addDisambiguation(label) + ")"
						+ "}";*/
				/*String queryString = prefixes
						+ "SELECT DISTINCT ?t WHERE { ?x rdfs:label " + label + ".\n"
						+ "?x <http://dbpedia.org/ontology/wikiPageDisambiguates> ?y.\n"
						+ "?y rdf:type/rdfs:subClassOf* ?t .\n"
						+ "}";*/
				String queryString = prefixes
						+ "SELECT DISTINCT ?t WHERE { "
						
						+ "{?x rdfs:label " + label + ".\n"
						+ "?x <http://dbpedia.org/ontology/wikiPageDisambiguates> ?y.\n"
						+ "?y rdf:type ?t .}\n"
						+ "UNION \n"
						+ "{?x rdfs:label " + addDisambiguation(label) + ".\n"
						+ "?x <http://dbpedia.org/ontology/wikiPageDisambiguates> ?y.\n"
						+ "?y rdf:type ?t .}\n"
						
						+ "}";
				Query query = QueryFactory.create(queryString);
				QueryExecution qexec = QueryExecutionFactory.create(query, model);
				LARQ.setDefaultIndex(qexec.getContext(), index);

				ResultSet rs = qexec.execSelect();
				try {
					while (rs.hasNext()) {

						QuerySolution qs = rs.next();
						result.add(qs.getResource("t").getURI());
					}
				}finally {
					qexec.close();
				}
			}catch(QueryParseException e){
				//System.err.println("Query parse exception, ok to skip");
			}
			catch(Exception e){
				System.out.println(e.toString());
			} finally {
				dataset.end();
			}
			
		}

		return result;
	}
	/**
	 * 
	 * @param cell
	 * @return
	 */
	public Set<String> getTypesOfEntitiesWithLabel(Cell cell) {
		
		Set<String> result = new HashSet<String>();

		String label = cell.getValue();
		String type = cell.getType();
		if(!type.equalsIgnoreCase("String"))
			return result;
		else{
			label = "\"" + label + "\"" + KnowledgeDatabaseConfig.languageTag ;
		}
			
		if(label.equals(""))
			return result;
		dataset.begin(ReadWrite.READ);
		try {
			String queryString = prefixes
					+ "SELECT DISTINCT ?t WHERE { ?x rdfs:label " + label + ".\n"
					+ "?x rdf:type/rdfs:subClassOf* ?t }";
			Query query = QueryFactory.create(queryString);
			QueryExecution qexec = QueryExecutionFactory.create(query, model);
			LARQ.setDefaultIndex(qexec.getContext(), index);

			ResultSet rs = qexec.execSelect();
			try {
				while (rs.hasNext()) {

					QuerySolution qs = rs.next();
					result.add(qs.getResource("t").getURI());

				}
			}finally {
				qexec.close();
			}
		}catch(QueryParseException e){
			//System.err.println("Query parse exception, ok to skip");
		}
		catch(Exception e){
			System.out.println(e.toString());
		} finally {
			dataset.end();
		}
		
		if(KnowledgeDatabaseConfig.dataDirectoryBase.toLowerCase().contains("dbpedia")){
			dataset.begin(ReadWrite.READ);
			try {
				/*String queryString = prefixes
						+ "SELECT DISTINCT ?t WHERE { ?x rdfs:label ?l" + ".\n"
						+ "?x <http://dbpedia.org/ontology/wikiPageDisambiguates> ?y.\n"
						+ "?y rdf:type/rdfs:subClassOf* ?t .\n"
						+ "FILTER( ?l = " + label + " || ?l = "  + addDisambiguation(label) + ")"
						+ "}";*/
				/*String queryString = prefixes
						+ "SELECT DISTINCT ?t WHERE { ?x rdfs:label " + label + ".\n"
						+ "?x <http://dbpedia.org/ontology/wikiPageDisambiguates> ?y.\n"
						+ "?y rdf:type/rdfs:subClassOf* ?t .\n"
						+ "}";*/
				String queryString = prefixes
						+ "SELECT DISTINCT ?t WHERE { "
						
						+ "{?x rdfs:label " + label + ".\n"
						+ "?x <http://dbpedia.org/ontology/wikiPageDisambiguates> ?y.\n"
						+ "?y rdf:type/rdfs:subClassOf* ?t .}\n"
						+ "UNION \n"
						+ "{?x rdfs:label " + addDisambiguation(label) + ".\n"
						+ "?x <http://dbpedia.org/ontology/wikiPageDisambiguates> ?y.\n"
						+ "?y rdf:type/rdfs:subClassOf* ?t .}\n"
						
						+ "}";
				Query query = QueryFactory.create(queryString);
				QueryExecution qexec = QueryExecutionFactory.create(query, model);
				LARQ.setDefaultIndex(qexec.getContext(), index);

				ResultSet rs = qexec.execSelect();
				try {
					while (rs.hasNext()) {

						QuerySolution qs = rs.next();
						result.add(qs.getResource("t").getURI());
					}
				}finally {
					qexec.close();
				}
			}catch(QueryParseException e){
				//System.err.println("Query parse exception, ok to skip");
			}
			catch(Exception e){
				System.out.println(e.toString());
			} finally {
				dataset.end();
			}
			
		}

		return result;
	}
	/**
	 * @deprecated User  getTypesOfEntitiesWithLabel(String cell) instead
	 * @param label
	 * @return
	 */
	public Set<String> getTypesOfEntitiesWithLabel(String label) {
		Set<String> result = new HashSet<String>();
		if(label.equals(""))
			return result;
		dataset.begin(ReadWrite.READ);
		try {
			String queryString = prefixes
					+ "SELECT DISTINCT ?x ?t WHERE { ?x rdfs:label \"" + label
					+ "\"" + KnowledgeDatabaseConfig.languageTag + "; \n"
					+ "rdf:type/rdfs:subClassOf* ?t }";
			Query query = QueryFactory.create(queryString);
			QueryExecution qexec = QueryExecutionFactory.create(query, model);
			LARQ.setDefaultIndex(qexec.getContext(), index);

			ResultSet rs = qexec.execSelect();
			try {
				while (rs.hasNext()) {

					QuerySolution qs = rs.next();

					// System.err.println(qs.getResource("x") + " |#| " +
					// qs.getResource("t"));
					result.add(qs.getResource("t").getURI());

				}
			}finally {
				qexec.close();
			}
		}catch(QueryParseException e){
			//System.err.println("Query parse exception, ok to skip");
		}
		catch(Exception e){
			System.out.println(e.toString());
		} finally {
			dataset.end();
		}

		return result;
	}

	/*
	 * public boolean isFuzzyMatch(String sourceLabel, String targetLabel,
	 * String ){
	 * 
	 * }
	 */

	/**
	 * 
	 * @param label
	 * @param score
	 *            Matching score
	 * @return
	 */
	public Set<String> getTypesOfEntitiesWithLabel(String label, double minScore) {
		Set<String> result = new HashSet<String>();
		if(label.equals(""))
			return result;
		dataset.begin(ReadWrite.READ);
		try {
			String queryString = prefixes + "SELECT DISTINCT ?x ?t WHERE {"
					+ "?lit pf:textMatch ('" + prepareForLucene(label) + "' "
					+ minScore + ") .\n" + "?x rdfs:label ?lit .\n"
					+ "?x rdf:type/rdfs:subClassOf* ?t }";

			//
			Query query = QueryFactory.create(queryString);
			QueryExecution qexec = QueryExecutionFactory.create(query, model);
			LARQ.setDefaultIndex(qexec.getContext(), index);

			ResultSet rs = qexec.execSelect();
			try {
				while (rs.hasNext()) {

					QuerySolution qs = rs.next();

					// System.err.println(qs.getResource("x") + " |#| " +
					// qs.getResource("t"));
					result.add(qs.getResource("t").getURI());

				}
			} finally {
				qexec.close();
			}
		} finally {
			dataset.end();
		}
		
		return result;
	}

	
	/**
	 * 
	 * @param label
	 * @param score
	 *            Matching score
	 * @return
	 */
	public Set<String> getTypesOfEntitiesWithLabel(String label, int maxMatches) {
		Set<String> result = new HashSet<String>();

		if(label.equals(""))
			return result;
		
		dataset.begin(ReadWrite.READ);
		try {
			String queryString = prefixes + "SELECT DISTINCT ?x ?t WHERE {"
					+ "?lit pf:textMatch ('" + prepareForLucene(label) + "' "
					+ maxMatches + ") .\n" + "?x rdfs:label ?lit .\n"
					+ "?x rdf:type/rdfs:subClassOf* ?t }";
			
//			String queryString = prefixes
//					+ "SELECT DISTINCT ?x ?t WHERE {" + "?lit pf:textMatch ('" + prepareForLucene(label) + "' " + 10 + ") .\n"
//					+ " ?x rdfs:label ?lit .\n"+"?x rdf:type/rdfs:subClassOf* <http://yago-knowledge.org/resource/wordnet_person_100007846> }";
		
			System.out.println(queryString);
			Query query = QueryFactory.create(queryString);
			QueryExecution qexec = QueryExecutionFactory.create(query, model);
			LARQ.setDefaultIndex(qexec.getContext(), index);

			ResultSet rs = qexec.execSelect();
			try {
				
				while (rs.hasNext()) {

					QuerySolution qs = rs.next();

					// System.err.println(qs.getResource("x") + " |#| " +
					// qs.getResource("t"));
					result.add(qs.getResource("t").getURI());

				}
			} finally {
				qexec.close();
			}
		} finally {
			dataset.end();
		}
		return result;
	}
	/**
	 * Get the URI mapping of the label, the type of the URI should be typeURI
	 * @param label
	 * @param typeURI
	 * @return
	 */
	public Set<String> getLabelURIwithTypeURI(String label, String typeURI) {
		Set<String> result = new HashSet<String>();
		if(label.equals(""))
			return result;
		dataset.begin(ReadWrite.READ);
		try {
			String queryString = prefixes
					+ "SELECT DISTINCT ?x WHERE { ?x rdfs:label \"" + label
					+ "\"" + KnowledgeDatabaseConfig.languageTag + "; \n"
					+ "rdf:type/rdfs:subClassOf*" + typeURI + "}";

			//

			Query query = QueryFactory.create(queryString);
			QueryExecution qexec = QueryExecutionFactory.create(query, model);
			LARQ.setDefaultIndex(qexec.getContext(), index);

			ResultSet rs = qexec.execSelect();
			try {
				while (rs.hasNext()) {

					QuerySolution qs = rs.next();

					// System.err.println(qs.getResource("x") + " |#| " +
					// qs.getResource("t"));
					result.add(qs.getResource("x").getURI());

				}
			} finally {
				qexec.close();
			}
		} finally {
			dataset.end();
		}

		return result;
	}
	
	/**
	 * Get the URI mapping of the label, the type of the URI should be typeURI
	 * @param label
	 * @param typeURI
	 * @return
	 */
	public Set<String> getLabelURIwithTypeURI(String label, String typeURI, int maxMatches) {
		Set<String> result = new HashSet<String>();
		if(label.equals(""))
			return result;
		dataset.begin(ReadWrite.READ);
		try {
			String queryString = prefixes + "SELECT DISTINCT ?x WHERE {"
					+ "?lit pf:textMatch ('" + prepareForLucene(label) + "' "
					+ maxMatches + ") .\n" + "?x rdfs:label ?lit .\n"
					+ "?x rdf:type/rdfs:subClassOf*" + typeURI +  "}";

			Query query = QueryFactory.create(queryString);
			QueryExecution qexec = QueryExecutionFactory.create(query, model);
			LARQ.setDefaultIndex(qexec.getContext(), index);

			ResultSet rs = qexec.execSelect();
			try {
				while (rs.hasNext()) {

					QuerySolution qs = rs.next();
					
					// System.err.println(qs.getResource("x") + " |#| " +
					// qs.getResource("t"));
					result.add(qs.getResource("x").getURI());

				}
			} finally {
				qexec.close();
			}
		} finally {
			dataset.end();
		}

		return result;
	}
	public Set<String> getEntitiesWithLabel(String label) {
		Set<String> result = new HashSet<String>();
		if(label.equals(""))
			return result;
		dataset.begin(ReadWrite.READ);
		try {
			String queryString = prefixes
					+ "SELECT DISTINCT ?x WHERE { ?x rdfs:label \"" + label
					+ "\"" + KnowledgeDatabaseConfig.languageTag + "}";

			Query query = QueryFactory.create(queryString);
			QueryExecution qexec = QueryExecutionFactory.create(query, model);
			LARQ.setDefaultIndex(qexec.getContext(), index);

			ResultSet rs = qexec.execSelect();
			try {
				while (rs.hasNext()) {
					QuerySolution qs = rs.next();

					result.add(qs.getResource("x").getURI());
				}
			} finally {
				qexec.close();
			}
		} finally {
			dataset.end();
		}

		return result;
	}

	/**
	 * 
	 * @param label
	 *            the label to match
	 * @param maxMatches
	 * @return URIs of entities having a label matching the specified label
	 */
	public Set<String> getEntitiesWithLabelMatching(String label, int maxMatches) {
		Set<String> result = new HashSet<String>();
		if(label.equals(""))
			return result;
		dataset.begin(ReadWrite.READ);

		String queryString = null;
		Query query = null;
		try {
			queryString = prefixes + "SELECT DISTINCT ?x WHERE { \n"
					+ "?lit pf:textMatch ('" + prepareForLucene(label) + "' "
					+ maxMatches + ") .\n" + "?x rdfs:label ?lit}";

			query = QueryFactory.create(queryString);
			QueryExecution qexec = QueryExecutionFactory.create(query, model);
			LARQ.setDefaultIndex(qexec.getContext(), index);

			ResultSet rs = qexec.execSelect();
			try {
				while (rs.hasNext()) {
					QuerySolution qs = rs.next();

					result.add(qs.getResource("x").getURI());
				}
			} catch (Exception ex) {

				System.out.println();
				System.out.println(query);
				ex.printStackTrace();
			} finally {
				qexec.close();
			}
		} catch (Exception ex) {

			System.out.println();
			System.out.println(query);
			ex.printStackTrace();

		} finally {
			dataset.end();
		}

		return result;
	}
	public String addDisambiguation(String val){
		StringBuilder sb = new StringBuilder(val);
		if (val.lastIndexOf("\"") != -1) {
		sb.insert(val.lastIndexOf("\""), " (disambiguation)");
		}else {
			sb.append(" (disambiguation)");
		}
		return new String(sb);
	}
	
	public Set<String> getDirectRelationShips(Cell cell1, Cell cell2, boolean withSuperProps) {
		String val1 = cell1.getValue().replace("\n", "");
		String type1 = cell1.getType();
		String val2 = cell2.getValue().replace("\n", "");
		String type2 = cell2.getType();
		
		Set<String> result = new HashSet<String>();
		if(val1.equals("")||val2.equals(""))
			return result;
		
		if(type1.equalsIgnoreCase("String")){
			val1 = "\"" + val1 + "\"" + KnowledgeDatabaseConfig.languageTag;
		}else{
			//numerical type, do not change anything
			//The first value is numerical type, cannot have any rels
			return result;
		}
		
		if(type2.equalsIgnoreCase("String")){
			val2 = "\"" + val2 + "\"" + KnowledgeDatabaseConfig.languageTag;
		}else{
			//numerical type, do not change anything
		}
		
		// First: cases with 1 free node

		dataset.begin(ReadWrite.READ);
		try {/*
			String queryString = prefixes
					+ "SELECT DISTINCT ?p2 WHERE { ?x rdfs:label \"" + val1
					+ "\"" + KnowledgeDatabaseConfig.languageTag + ".\n"
					+ " ?x ?p2 \"" + val2 + "\""
					+ "^^xsd:decimal"
					//+ KnowledgeDatabaseConfig.languageTag 
					+ "}";*/
	
			String queryString = prefixes
					+ "SELECT DISTINCT ?p2 WHERE { ?x rdfs:label " + val1 + ".\n"
					+ " ?x ?p2 " + val2 
					+ "}";

			Query query = QueryFactory.create(queryString);
			QueryExecution qexec = QueryExecutionFactory.create(query, model);
			LARQ.setDefaultIndex(qexec.getContext(), index);

			ResultSet rs = qexec.execSelect();
			try {
				while (rs.hasNext()) {
					QuerySolution qs = rs.next();

					// if(isLabelProperty(qs.getResource("p1")))
					// {
					// result.add(qs.getResource("p2").getURI());
					// }
					// else if(isLabelProperty(qs.getResource("p2")))
					// {
					// result.add(qs.getResource("p1").getURI());
					// }

					result.add(qs.getResource("p2").getURI());
				}
			} finally {
				qexec.close();
			}

		} catch(QueryParseException e){
			//System.err.println("Query parse exception, ok to skip");
		}
		catch(Exception e){
			System.out.println(e.toString());
		}finally {
			dataset.end();
		}
		
		// First: cases with 1 free node, add wiki-disambiguate
		if(KnowledgeDatabaseConfig.dataDirectoryBase.toLowerCase().contains("dbpedia")){
			dataset.begin(ReadWrite.READ);
			try {
				String queryString = prefixes
						+ "SELECT DISTINCT ?p2 WHERE {" 
						
						+ " {?x rdfs:label " + val1 + ".\n"
						+ "?x <http://dbpedia.org/ontology/wikiPageDisambiguates> ?y.\n"
						+ " ?y ?p2 " + val2 + "} \n"
						
						+ "UNION \n"
						
						+ " {?x rdfs:label " + addDisambiguation(val1)+ ".\n"
						+ "?x <http://dbpedia.org/ontology/wikiPageDisambiguates> ?y.\n"
						+ " ?y ?p2 " + val2 + "} \n"
						
						+ "}";

				Query query = QueryFactory.create(queryString);
				QueryExecution qexec = QueryExecutionFactory.create(query, model);
				LARQ.setDefaultIndex(qexec.getContext(), index);

				ResultSet rs = qexec.execSelect();
				try {
					while (rs.hasNext()) {
						QuerySolution qs = rs.next();
						result.add(qs.getResource("p2").getURI());
						
					}
				} finally {
					qexec.close();
				}

			} catch(QueryParseException e){
				//System.err.println("Query parse exception, ok to skip");
			}
			catch(Exception e){
				System.out.println(e.toString());
			}finally {
				dataset.end();
			}
		}
		// Second: cases with 2 freeNodes
		dataset.begin(ReadWrite.READ);
		try {
			String queryString = prefixes
					+ "SELECT DISTINCT ?p WHERE {?x ?p ?y .\n"
					+ " ?x rdfs:label " + val1 + ".\n"
					+ " ?y rdfs:label " + val2 +  "}";
			Query query = QueryFactory.create(queryString);
			QueryExecution qexec = QueryExecutionFactory.create(query, model);
			LARQ.setDefaultIndex(qexec.getContext(), index);
			
			
			
			ResultSet rs = qexec.execSelect();
			try {
				while (rs.hasNext()) {
					QuerySolution qs = rs.next();

					// if(isLabelProperty(qs.getResource("p1")) &&
					// isLabelProperty(qs.getResource("p2")) )
					// {
					
					
					result.add(qs.getResource("p").getURI());
					
					
					// }
				}
			} finally {
				qexec.close();
			}
		} catch(QueryParseException e){
			//System.err.println("Query parse exception, ok to skip");
		}
		catch(Exception e){
			System.out.println(e.toString());
		}finally {
			dataset.end();
		}
		if(KnowledgeDatabaseConfig.dataDirectoryBase.toLowerCase().contains("dbpedia")){
			
			//both add disamgiguation
			if(type1.equalsIgnoreCase("String") && type2.equalsIgnoreCase("String")){
				dataset.begin(ReadWrite.READ);
				try {
					String queryString = prefixes
							+ "SELECT DISTINCT ?p WHERE {" 
							
							+ "{?a ?p ?b .\n"
							+ " ?x rdfs:label " + addDisambiguation(val1)+ ".\n"
							+ "?x <http://dbpedia.org/ontology/wikiPageDisambiguates> ?a.\n"
							+ "?y <http://dbpedia.org/ontology/wikiPageDisambiguates> ?b.\n"
							+ " ?y rdfs:label " + addDisambiguation(val2) + "}\n"
							
							+ "UNION \n"
							
							+ "{?a ?p ?b .\n"
							+ " ?x rdfs:label " + val1+ ".\n"
							+ "?x <http://dbpedia.org/ontology/wikiPageDisambiguates> ?a.\n"
							+ "?y <http://dbpedia.org/ontology/wikiPageDisambiguates> ?b.\n"
							+ " ?y rdfs:label " + addDisambiguation(val2) + "}\n"
							
							+ "UNION \n"
							
							+ "{?a ?p ?b .\n"
							+ " ?x rdfs:label " + addDisambiguation(val1)+ ".\n"
							+ "?x <http://dbpedia.org/ontology/wikiPageDisambiguates> ?a.\n"
							+ "?y <http://dbpedia.org/ontology/wikiPageDisambiguates> ?b.\n"
							+ " ?y rdfs:label " + val2 + "}\n"
							
							+ "UNION \n"
							
							+ "{?a ?p ?b .\n"
							+ " ?x rdfs:label " + val1+ ".\n"
							+ "?x <http://dbpedia.org/ontology/wikiPageDisambiguates> ?a.\n"
							+ "?y <http://dbpedia.org/ontology/wikiPageDisambiguates> ?b.\n"
							+ " ?y rdfs:label " + val2 + "}\n"
							
							
							
							+ "}";
					Query query = QueryFactory.create(queryString);
					QueryExecution qexec = QueryExecutionFactory.create(query, model);
					LARQ.setDefaultIndex(qexec.getContext(), index);
					
					
					
					ResultSet rs = qexec.execSelect();
					try {
						while (rs.hasNext()) {
							QuerySolution qs = rs.next();

							// if(isLabelProperty(qs.getResource("p1")) &&
							// isLabelProperty(qs.getResource("p2")) )
							// {
							
							
							result.add(qs.getResource("p").getURI());
							
							
							// }
						}
					} finally {
						qexec.close();
					}
				} catch(QueryParseException e){
					//System.err.println("Query parse exception, ok to skip");
				}
				catch(Exception e){
					System.out.println(e.toString());
				}finally {
					dataset.end();
				}
			}
			if(type1.equalsIgnoreCase("String")){
				//only first one
				dataset.begin(ReadWrite.READ);
				try {
					String queryString = prefixes
							+ "SELECT DISTINCT ?p WHERE {" 
							
							+ "{?a ?p ?b .\n"
							+ " ?x rdfs:label " + val1+ ".\n"
							+ "?x <http://dbpedia.org/ontology/wikiPageDisambiguates> ?a.\n"
							+ " ?b rdfs:label " + val2 + "}\n"
							
							+ "UNION \n"
							
							+ "{?a ?p ?b .\n"
							+ " ?x rdfs:label " +addDisambiguation(val1)+ ".\n"
							+ "?x <http://dbpedia.org/ontology/wikiPageDisambiguates> ?a.\n"
							+ " ?b rdfs:label " + val2 + "}\n"
							
							+  "}";
					Query query = QueryFactory.create(queryString);
					QueryExecution qexec = QueryExecutionFactory.create(query, model);
					LARQ.setDefaultIndex(qexec.getContext(), index);
					
					
					
					ResultSet rs = qexec.execSelect();
					try {
						while (rs.hasNext()) {
							QuerySolution qs = rs.next();

							// if(isLabelProperty(qs.getResource("p1")) &&
							// isLabelProperty(qs.getResource("p2")) )
							// {
							
							
							result.add(qs.getResource("p").getURI());
							
							
							// }
						}
					} finally {
						qexec.close();
					}
				} catch(QueryParseException e){
					//System.err.println("Query parse exception, ok to skip");
				}
				catch(Exception e){
					System.out.println(e.toString());
				}finally {
					dataset.end();
				}
			}
			if(type2.equalsIgnoreCase("String")){
				//only second one
				dataset.begin(ReadWrite.READ);
				try {

					String queryString = prefixes
							+ "SELECT DISTINCT ?p WHERE { "  
							
							+ "{?a ?p ?b .\n"
							+ " ?a rdfs:label " + val1 + ".\n"
							+ "?y <http://dbpedia.org/ontology/wikiPageDisambiguates> ?b.\n"
							+ " ?y rdfs:label " + val2  + "}\n"
							
							+ "UNION \n"
							
							+ "{?a ?p ?b .\n"
							+ " ?a rdfs:label " + val1 + ".\n"
							+ "?y <http://dbpedia.org/ontology/wikiPageDisambiguates> ?b.\n"
							+ " ?y rdfs:label " + addDisambiguation(val2)  + "}\n"
							
							
							
							+  "}";
					Query query = QueryFactory.create(queryString);
					QueryExecution qexec = QueryExecutionFactory.create(query, model);
					LARQ.setDefaultIndex(qexec.getContext(), index);
					
					
					
					ResultSet rs = qexec.execSelect();
					try {
						while (rs.hasNext()) {
							QuerySolution qs = rs.next();

							// if(isLabelProperty(qs.getResource("p1")) &&
							// isLabelProperty(qs.getResource("p2")) )
							// {
							
							
							result.add(qs.getResource("p").getURI());
							
							
							// }
						}
					} finally {
						qexec.close();
					}
				} catch(QueryParseException e){
					//System.err.println("Query parse exception, ok to skip");
				}
				catch(Exception e){
					System.out.println(e.toString());
				}finally {
					dataset.end();
				}
			}
		
		}
		if (withSuperProps) {
			// Get superproperties as well
			HashSet<String> superProps = new HashSet<>();
			for (String p : result) {
				dataset.begin(ReadWrite.READ);
				try {
					String queryString2 = prefixes
							+ "SELECT DISTINCT ?p WHERE { <" + URIref.encode(p)
							+ "> rdfs:subPropertyOf+ ?p }";

					Query query2 = QueryFactory.create(queryString2);
					QueryExecution qexec2 = QueryExecutionFactory.create(
							query2, model);
					LARQ.setDefaultIndex(qexec2.getContext(), index);

					ResultSet rs2 = qexec2.execSelect();
					try {
						while (rs2.hasNext()) {
							QuerySolution qs2 = rs2.next();

							superProps.add(qs2.getResource("p").getURI());
						}
					} finally {
						qexec2.close();
					}
				} catch(QueryParseException e){
					//System.err.println("Query parse exception, ok to skip");
				}
				catch(Exception e){
					System.out.println(e.toString());
				}finally {
					dataset.end();
				}
			}

			result.addAll(superProps);
		}
		return result;
	}
	/**
	 * @deprecated Use getDirectRelationShips(Cell cell1, Cell cell2) instead!
	 * @param val1
	 * @param val2
	 * @param withSuperProps
	 * @return
	 */
	public Set<String> getDirectRelationShips(String val1, String val2,
			boolean withSuperProps) {
		val1 = val1.replace("\n", "");
		val2 = val2.replace("\n", "");

		Set<String> result = new HashSet<String>();
		if(val1.equals("")||val2.equals(""))
			return result;
		// First: cases with 1 free node

		dataset.begin(ReadWrite.READ);
		try {/*
			String queryString = prefixes
					+ "SELECT DISTINCT ?p2 WHERE { ?x rdfs:label \"" + val1
					+ "\"" + KnowledgeDatabaseConfig.languageTag + ".\n"
					+ " ?x ?p2 \"" + val2 + "\""
					+ "^^xsd:decimal"
					//+ KnowledgeDatabaseConfig.languageTag 
					+ "}";*/
			
			String queryString = prefixes
					+ "SELECT DISTINCT ?p2 WHERE { ?x rdfs:label \"" + val1
					+ "\"" + KnowledgeDatabaseConfig.languageTag + ".\n"
					+ " ?x ?p2 \"" + val2 + "\""
					+ KnowledgeDatabaseConfig.languageTag 
					+ "}";

			Query query = QueryFactory.create(queryString);
			QueryExecution qexec = QueryExecutionFactory.create(query, model);
			LARQ.setDefaultIndex(qexec.getContext(), index);

			ResultSet rs = qexec.execSelect();
			try {
				while (rs.hasNext()) {
					QuerySolution qs = rs.next();

					// if(isLabelProperty(qs.getResource("p1")))
					// {
					// result.add(qs.getResource("p2").getURI());
					// }
					// else if(isLabelProperty(qs.getResource("p2")))
					// {
					// result.add(qs.getResource("p1").getURI());
					// }

					result.add(qs.getResource("p2").getURI());
				}
			} finally {
				qexec.close();
			}

		} catch(QueryParseException e){
			//System.err.println("Query parse exception, ok to skip");
		}
		catch(Exception e){
			System.out.println(e.toString());
		}finally {
			dataset.end();
		}

		// Second: cases with 2 freeNodes
		dataset.begin(ReadWrite.READ);
		try {
			String queryString = prefixes
					+ "SELECT DISTINCT ?p WHERE {?x ?p ?y .\n"
					+ " ?x rdfs:label \"" + val1 + "\""
					+ KnowledgeDatabaseConfig.languageTag + ".\n"
					+ " ?y rdfs:label \"" + val2 + "\""
					+ KnowledgeDatabaseConfig.languageTag + "}";
			Query query = QueryFactory.create(queryString);
			QueryExecution qexec = QueryExecutionFactory.create(query, model);
			LARQ.setDefaultIndex(qexec.getContext(), index);
			
			
			
			ResultSet rs = qexec.execSelect();
			try {
				while (rs.hasNext()) {
					QuerySolution qs = rs.next();

					// if(isLabelProperty(qs.getResource("p1")) &&
					// isLabelProperty(qs.getResource("p2")) )
					// {
					
					
					result.add(qs.getResource("p").getURI());
					
					
					// }
				}
			} finally {
				qexec.close();
			}
		} catch(QueryParseException e){
			//System.err.println("Query parse exception, ok to skip");
		}
		catch(Exception e){
			System.out.println(e.toString());
		}finally {
			dataset.end();
		}

		if (withSuperProps) {
			// Get superproperties as well
			HashSet<String> superProps = new HashSet<>();
			for (String p : result) {
				dataset.begin(ReadWrite.READ);
				try {
					String queryString2 = prefixes
							+ "SELECT DISTINCT ?p WHERE { <" + URIref.encode(p)
							+ "> rdfs:subPropertyOf+ ?p }";

					Query query2 = QueryFactory.create(queryString2);
					QueryExecution qexec2 = QueryExecutionFactory.create(
							query2, model);
					LARQ.setDefaultIndex(qexec2.getContext(), index);

					ResultSet rs2 = qexec2.execSelect();
					try {
						while (rs2.hasNext()) {
							QuerySolution qs2 = rs2.next();

							superProps.add(qs2.getResource("p").getURI());
						}
					} finally {
						qexec2.close();
					}
				} catch(QueryParseException e){
					//System.err.println("Query parse exception, ok to skip");
				}
				catch(Exception e){
					System.out.println(e.toString());
				}finally {
					dataset.end();
				}
			}

			result.addAll(superProps);
		}
		return result;
	}

	/**
	 * Approximate Matches, Need to be changed
	 * @param val1
	 * @param val2
	 * @param withSuperProps
	 * @param maxMatches
	 * @return
	 */
	public Set<String> getDirectRelationShips(String val1, String val2,
			boolean withSuperProps, int maxMatches) {
		val1 = val1.replace("\n", "");
		val2 = val2.replace("\n", "");

		Set<String> result = new HashSet<String>();
		if(val1.equals("")||val2.equals(""))
			return result;
		// First: cases with 1 free node

		dataset.begin(ReadWrite.READ);
		try {
			String queryString = prefixes + "SELECT DISTINCT ?p2 WHERE { "
					+ "?lit1 pf:textMatch ('" + prepareForLucene(val1) + "' "
					+ maxMatches + ") .\n" + "?lit2 pf:textMatch ('"
					+ prepareForLucene(val2) + "' " + maxMatches + ") .\n"
					+ "?x rdfs:label ?lit1 " + ".\n" + " ?x ?p2 ?lit2 " + "}";

			Query query = QueryFactory.create(queryString);
			QueryExecution qexec = QueryExecutionFactory.create(query, model);
			LARQ.setDefaultIndex(qexec.getContext(), index);

			ResultSet rs = qexec.execSelect();
			try {
				while (rs.hasNext()) {
					QuerySolution qs = rs.next();

					// if(isLabelProperty(qs.getResource("p1")))
					// {
					// result.add(qs.getResource("p2").getURI());
					// }
					// else if(isLabelProperty(qs.getResource("p2")))
					// {
					// result.add(qs.getResource("p1").getURI());
					// }

					result.add(qs.getResource("p2").getURI());
				}
			} finally {
				qexec.close();
			}

		} finally {
			dataset.end();
		}

		// Second: cases with 2 freeNodes
		dataset.begin(ReadWrite.READ);
		try {
			String queryString = prefixes + "SELECT DISTINCT ?p WHERE {"
					+ "?lit1 pf:textMatch ('" + prepareForLucene(val1) + "' "
					+ maxMatches + ") .\n" + "?lit2 pf:textMatch ('"
					+ prepareForLucene(val2) + "' " + maxMatches + ") .\n"
					+ "?x ?p ?y .\n" + " ?x rdfs:label ?lit1 " + ".\n"
					+ " ?y rdfs:label ?lit2 " + "}";
			Query query = QueryFactory.create(queryString);
			QueryExecution qexec = QueryExecutionFactory.create(query, model);
			LARQ.setDefaultIndex(qexec.getContext(), index);

			ResultSet rs = qexec.execSelect();
			try {
				while (rs.hasNext()) {
					QuerySolution qs = rs.next();

					// if(isLabelProperty(qs.getResource("p1")) &&
					// isLabelProperty(qs.getResource("p2")) )
					// {
					result.add(qs.getResource("p").getURI());
					// }
				}
			} finally {
				qexec.close();
			}
		} finally {
			dataset.end();
		}

		if (withSuperProps) {
			// Get superproperties as well
			HashSet<String> superProps = new HashSet<>();
			for (String p : result) {
				dataset.begin(ReadWrite.READ);
				try {
					String queryString2 = prefixes
							+ "SELECT DISTINCT ?p WHERE { <" + URIref.encode(p)
							+ "> rdfs:subPropertyOf+ ?p }";

					Query query2 = QueryFactory.create(queryString2);
					QueryExecution qexec2 = QueryExecutionFactory.create(
							query2, model);
					LARQ.setDefaultIndex(qexec2.getContext(), index);

					ResultSet rs2 = qexec2.execSelect();
					try {
						while (rs2.hasNext()) {
							QuerySolution qs2 = rs2.next();

							superProps.add(qs2.getResource("p").getURI());
						}
					} finally {
						qexec2.close();
					}
				} finally {
					dataset.end();
				}
			}

			result.addAll(superProps);
		}
		return result;
	}
	/**
	 * Two cells are connected by exactly one node
	 * @param cell1
	 * @param cell2
	 * @param withSuperProps
	 * @return
	 */
	public Set<String> getInDirectRelationShipsHop1(Cell cell1, Cell cell2, boolean withSuperProps) {
		String val1 = cell1.getValue().replace("\n", "");
		String type1 = cell1.getType();
		String val2 = cell2.getValue().replace("\n", "");
		String type2 = cell2.getType();
		
		Set<String> result = new HashSet<String>();
		if(val1.equals("")||val2.equals(""))
			return result;
		
		if(type1.equalsIgnoreCase("String")){
			val1 = "\"" + val1 + "\"" + KnowledgeDatabaseConfig.languageTag;
		}else{
			//numerical type, do not change anything
			//The first value is numerical type, cannot have any rels
			return result;
		}
		
		if(type2.equalsIgnoreCase("String")){
			val2 = "\"" + val2 + "\"" + KnowledgeDatabaseConfig.languageTag;
		}else{
			//numerical type, do not change anything
		}
		
		// First: cases with 1 free node

		dataset.begin(ReadWrite.READ);
		try {/*
			String queryString = prefixes
					+ "SELECT DISTINCT ?p2 WHERE { ?x rdfs:label \"" + val1
					+ "\"" + KnowledgeDatabaseConfig.languageTag + ".\n"
					+ " ?x ?p2 \"" + val2 + "\""
					+ "^^xsd:decimal"
					//+ KnowledgeDatabaseConfig.languageTag 
					+ "}";*/
			
			String queryString = prefixes
					+ "SELECT DISTINCT ?p1 ?p2 WHERE { ?x rdfs:label " + val1 + ".\n"
					+ " ?x ?p1 ?y " + ".\n"
					+ " ?y ?p2 " + val2 
					+ "}";

			Query query = QueryFactory.create(queryString);
			QueryExecution qexec = QueryExecutionFactory.create(query, model);
			LARQ.setDefaultIndex(qexec.getContext(), index);

			ResultSet rs = qexec.execSelect();
			try {
				while (rs.hasNext()) {
					QuerySolution qs = rs.next();

					// if(isLabelProperty(qs.getResource("p1")))
					// {
					// result.add(qs.getResource("p2").getURI());
					// }
					// else if(isLabelProperty(qs.getResource("p2")))
					// {
					// result.add(qs.getResource("p1").getURI());
					// }
					String temp = qs.getResource("p1").getURI() +","+ qs.getResource("p2").getURI();
					result.add(temp);
				}
			} finally {
				qexec.close();
			}

		} catch(QueryParseException e){
			//System.err.println("Query parse exception, ok to skip");
		}
		catch(Exception e){
			System.out.println(e.toString());
		}finally {
			dataset.end();
		}
		
		// First: cases with 1 free node, add wiki-disambiguate
		if(KnowledgeDatabaseConfig.dataDirectoryBase.toLowerCase().contains("dbpedia")){
			dataset.begin(ReadWrite.READ);
			try {
				String queryString = prefixes
						+ "SELECT DISTINCT ?p1 ?p2 WHERE { ?x rdfs:label " + val1 + ".\n"
						+ "?x <http://dbpedia.org/ontology/wikiPageDisambiguates> ?y.\n"
						+ " ?y ?p1 ?z " + ".\n"
						+ " ?z ?p2 " + val2 
						+ "}";

				Query query = QueryFactory.create(queryString);
				QueryExecution qexec = QueryExecutionFactory.create(query, model);
				LARQ.setDefaultIndex(qexec.getContext(), index);

				ResultSet rs = qexec.execSelect();
				try {
					while (rs.hasNext()) {
						QuerySolution qs = rs.next();
						String temp = qs.getResource("p1").getURI() + ","+qs.getResource("p2").getURI();
						result.add(temp);
					}
				} finally {
					qexec.close();
				}

			} catch(QueryParseException e){
				//System.err.println("Query parse exception, ok to skip");
			}
			catch(Exception e){
				System.out.println(e.toString());
			}finally {
				dataset.end();
			}
		}
		// Second: cases with 2 freeNodes
		dataset.begin(ReadWrite.READ);
		try {
			String queryString = prefixes
					+ "SELECT DISTINCT ?p1 ?p2 WHERE {?x ?p1 ?z .\n"
					+ " ?z ?p2 ?y " +  ".\n"
					+ " ?x rdfs:label " + val1 + ".\n"
					+ " ?y rdfs:label " + val2 +  "}";
			Query query = QueryFactory.create(queryString);
			QueryExecution qexec = QueryExecutionFactory.create(query, model);
			LARQ.setDefaultIndex(qexec.getContext(), index);
			
			
			
			ResultSet rs = qexec.execSelect();
			try {
				while (rs.hasNext()) {
					QuerySolution qs = rs.next();

					// if(isLabelProperty(qs.getResource("p1")) &&
					// isLabelProperty(qs.getResource("p2")) )
					// {
					
					
					String temp = qs.getResource("p1").getURI() + ","+qs.getResource("p2").getURI();
					result.add(temp);
					
					
					// }
				}
			} finally {
				qexec.close();
			}
		} catch(QueryParseException e){
			//System.err.println("Query parse exception, ok to skip");
		}
		catch(Exception e){
			System.out.println(e.toString());
		}finally {
			dataset.end();
		}
		if(KnowledgeDatabaseConfig.dataDirectoryBase.toLowerCase().contains("dbpedia")){
			dataset.begin(ReadWrite.READ);
			try {
				String queryString = prefixes
						+ "SELECT DISTINCT ?p1 ?p2 WHERE {?a ?p1 ?c .\n"
						+ " ?c ?p2 ?b " +  ".\n"
						+ " ?x rdfs:label " + val1 + ".\n"
						+ "?x <http://dbpedia.org/ontology/wikiPageDisambiguates> ?a.\n"
						+ "?y <http://dbpedia.org/ontology/wikiPageDisambiguates> ?b.\n"
						+ " ?y rdfs:label " + val2 +  "}";
				Query query = QueryFactory.create(queryString);
				QueryExecution qexec = QueryExecutionFactory.create(query, model);
				LARQ.setDefaultIndex(qexec.getContext(), index);
				
				
				
				ResultSet rs = qexec.execSelect();
				try {
					while (rs.hasNext()) {
						QuerySolution qs = rs.next();

						// if(isLabelProperty(qs.getResource("p1")) &&
						// isLabelProperty(qs.getResource("p2")) )
						// {
						
						
						String temp = qs.getResource("p1").getURI() + ","+qs.getResource("p2").getURI();
						result.add(temp);
						
						
						// }
					}
				} finally {
					qexec.close();
				}
			} catch(QueryParseException e){
				//System.err.println("Query parse exception, ok to skip");
			}
			catch(Exception e){
				System.out.println(e.toString());
			}finally {
				dataset.end();
			}
		}
		if (withSuperProps) {
			// Get superproperties as well
			HashSet<String> superProps = new HashSet<>();
			for (String p : result) {
				dataset.begin(ReadWrite.READ);
				try {
					String queryString2 = prefixes
							+ "SELECT DISTINCT ?p WHERE { <" + URIref.encode(p)
							+ "> rdfs:subPropertyOf+ ?p }";

					Query query2 = QueryFactory.create(queryString2);
					QueryExecution qexec2 = QueryExecutionFactory.create(
							query2, model);
					LARQ.setDefaultIndex(qexec2.getContext(), index);

					ResultSet rs2 = qexec2.execSelect();
					try {
						while (rs2.hasNext()) {
							QuerySolution qs2 = rs2.next();

							superProps.add(qs2.getResource("p").getURI());
						}
					} finally {
						qexec2.close();
					}
				} catch(QueryParseException e){
					//System.err.println("Query parse exception, ok to skip");
				}
				catch(Exception e){
					System.out.println(e.toString());
				}finally {
					dataset.end();
				}
			}

			result.addAll(superProps);
		}
		return result;
	}
	public Set<String> getAllTypes() {
		Set<String> result = new HashSet<String>();
		dataset.begin(ReadWrite.READ);
		try {
			String queryString = prefixes
					+ "SELECT DISTINCT ?t WHERE { ?x  rdf:type ?t}";
			
			/*String queryString = prefixes
					+ "SELECT DISTINCT ?t WHERE { ?x  rdf:type/rdfs:subClassOf* ?t}";*/
			
			Query query = QueryFactory.create(queryString);
			QueryExecution qexec = QueryExecutionFactory.create(query, model);
			LARQ.setDefaultIndex(qexec.getContext(), index);

			ResultSet rs = qexec.execSelect();
			try {
				while (rs.hasNext()) {
					result.add(rs.next().getResource("t").getURI());
				}
			} finally {
				qexec.close();
			}
		} finally {
			dataset.end();
		}

		return result;
	}

	public Set<String> getAllRelationships() {
		Set<String> result = new HashSet<String>();
		dataset.begin(ReadWrite.READ);
		try {
			String queryString = prefixes
					+ "SELECT DISTINCT ?x WHERE { ?x  rdf:type rdf:Property}";
			Query query = QueryFactory.create(queryString);
			QueryExecution qexec = QueryExecutionFactory.create(query, model);
			LARQ.setDefaultIndex(qexec.getContext(), index);

			ResultSet rs = qexec.execSelect();
			try {
				while (rs.hasNext()) {
					result.add(rs.next().getResource("x").getURI());
				}
			} finally {
				qexec.close();
			}
		} finally {
			dataset.end();
		}

		try {
			String queryString = prefixes
					+ "SELECT DISTINCT ?x WHERE { ?x  rdf:type owl:FunctionalProperty}";
			Query query = QueryFactory.create(queryString);
			QueryExecution qexec = QueryExecutionFactory.create(query, model);
			LARQ.setDefaultIndex(qexec.getContext(), index);

			ResultSet rs = qexec.execSelect();
			try {
				while (rs.hasNext()) {
					result.add(rs.next().getResource("x").getURI());
				}
			} finally {
				qexec.close();
			}
		} finally {
			dataset.end();
		}

		return result;
	}
	public Set<String> getAllRelationships_2() {
		Set<String> result = new HashSet<String>();
		dataset.begin(ReadWrite.READ);
		try {
			String queryString = prefixes
					+ "SELECT DISTINCT ?p WHERE { ?x  ?p ?y}";
			Query query = QueryFactory.create(queryString);
			QueryExecution qexec = QueryExecutionFactory.create(query, model);
			LARQ.setDefaultIndex(qexec.getContext(), index);

			ResultSet rs = qexec.execSelect();
			try {
				while (rs.hasNext()) {
					result.add(rs.next().getResource("p").getURI());
				}
			} finally {
				qexec.close();
			}
		} finally {
			dataset.end();
		}
		return result;
	}
	public String prepareForLucene(String s) {
		String processedVal = "";

		String v = LUCENE_PATTERN.matcher(s).replaceAll(REPLACEMENT_STRING);

		StringTokenizer tokenizer = new StringTokenizer(v);
		while (tokenizer.hasMoreTokens()) {
			processedVal += " +" + tokenizer.nextToken() + "~"
					+ FUZZINESS_THRESHOLD;
		}

		processedVal = processedVal.replace("'", "\\'");

		return processedVal;
	}

	@Override
	public void close() throws Exception {
		logger.debug("KBReader closed");
		TDB.sync(model);
		model.close();
		logger.debug("model closed");
		dataset.close();
		logger.debug("dataset closed");
		LARQ.getDefaultIndex().close();
		logger.debug("Index closed");
		TDB.closedown();
		logger.debug("TDB closed");
	}
	
	/*******************************
	 * ******************************
	 * THE FOLLOWINGS ARE METHODS USED BY SUNITA'S DISCOVERY ALGORITHM
	 * ******************************
	 * ******************************
	 * ******************************
	 * 
	 */
	
	public int getMinNumSteps(String value, String finalTypeURI) {

		int result = Integer.MAX_VALUE - 1000;

		dataset.begin(ReadWrite.READ);

		String queryString = null;
		try {
			queryString = prefixes
					+ "SELECT DISTINCT ?t WHERE { ?x rdfs:label \"" + value
					+ "\"" + KnowledgeDatabaseConfig.languageTag + "; \n"
					+ "rdf:type ?t}";

			Query query = QueryFactory.create(queryString);
			QueryExecution qexec = QueryExecutionFactory.create(query, model);
			LARQ.setDefaultIndex(qexec.getContext(), index);

			ResultSet rs = qexec.execSelect();
			try {
				while (rs.hasNext()) {

					QuerySolution qs = rs.next();

					String typeURI = qs.getResource("t").getURI();

					if (finalTypeURI.equals(typeURI)) {
						result = 1;

						break;
					} else {
						int steps = 1 + getMinTypeSteps(typeURI, finalTypeURI);
						if (steps < result) {
							result = steps;
						}
					}
				}
			} finally {
				qexec.close();
			}
		} catch (QueryParseException ex) {
			System.out.println("\nError:\n=======\n" + queryString);
			ex.printStackTrace();
		} finally {
			dataset.end();
		}

		return result;

	}

	public int getMinNumStepsFromEntityToType(String entityURI,
			String finalTypeURI) {

		int result = Integer.MAX_VALUE - 1000;

		if(!instanceChecking(entityURI,finalTypeURI))
			return result;
		
		HashSet<String> directTypes = getTypes(URIref.encode(entityURI), false);

		for (String typeURI : directTypes) {
			if (finalTypeURI.equals(typeURI)) {
				result = 1;

				break;
			} else {
				int steps = 1 + getMinTypeSteps(typeURI, finalTypeURI);
				if (steps < result) {
					result = steps;
				}
			}
		}

		return result;
	}

	public int getMinTypeSteps(String typeURI, String finalType) {
		return getMinStepsFollowingProperty(typeURI, finalType,
				"rdfs:subClassOf");
	}

	public int getMinNumSteps(String value1, String value2, String finalRel) {
		Set<String> subProps = getDirectRelationShips(value1, value2, false);

		int min = Integer.MAX_VALUE - 1000;

		for (String r : subProps) {
			if (r.equals(finalRel)) {
				return 1;
			} else {
				int steps = 1 + getMinRelSteps(r, finalRel);
				if (steps < min) {
					min = steps;
				}
			}
		}

		return min;
	}

	public int getMinRelSteps(String relURI, String finalRel) {
		return getMinStepsFollowingProperty(relURI, finalRel,
				"rdfs:subPropertyOf");
	}

	public int getMinStepsFollowingProperty(String uri1, String uri2,
			String property) {
		if (uri1.equals(uri2)) {
			return 0;
		}

		Queue<ArrayList<String>> paths = new LinkedList<>();

		ArrayList<String> firstPath = new ArrayList<>();
		firstPath.add(uri1);
		paths.add(firstPath);

		while (!paths.isEmpty()) {
			ArrayList<String> curPath = paths.remove();

			String queryString = null;
			try {
				queryString = prefixes + "SELECT DISTINCT ?x WHERE { <"
						+ URIref.encode(curPath.get(curPath.size() - 1)) + "> "
						+ property + " ?x}";

				Query query = QueryFactory.create(queryString);
				QueryExecution qexec = QueryExecutionFactory.create(query,
						model);
				LARQ.setDefaultIndex(qexec.getContext(), index);

				ResultSet rs = qexec.execSelect();
				try {
					while (rs.hasNext()) {
						QuerySolution qs = rs.next();
						String x = qs.getResource("x").getURI();
						if (curPath.contains(x)) {
							continue; // useless path
						}
						if (x.equals(uri2)) {
							qexec.close();
							dataset.end();
							return curPath.size(); // no need for -1 since we
													// did not add x
						} else {
							ArrayList<String> newPath = new ArrayList<>();
							newPath.addAll(curPath);
							newPath.add(x);
							
							paths.add(newPath);
						}
					}
				} finally {
					qexec.close();
				}
			} catch (QueryParseException ex) {
				System.out.println("\nError:\n=======\n" + queryString);
				ex.printStackTrace();
			} finally {
				dataset.end();
			}
		}

		return Integer.MAX_VALUE - 1000;
	}

	public Set<String> getRelationshipsBetweenTypes(String typeURI1,
			String typeURI2) {
		Set<String> result = new HashSet<String>();

		dataset.begin(ReadWrite.READ);
		try {
			String queryString = prefixes
					+ "SELECT DISTINCT ?sp WHERE {?p rdfs:subPropertyOf* ?sp.\n" // TODO:
																					// should
																					// we
																					// add
																					// superprops?
					+ "?p rdf:type rdf:Property ;\n" + "  rdfs:domain ?t1;\n"
					+ "  rdfs:range ?t2.\n" + " <" + typeURI1
					+ "> rdfs:subClassOf* ?t1.\n" + " <" + typeURI2
					+ "> rdfs:subClassOf* ?t2 }";

			Query query = QueryFactory.create(queryString);
			QueryExecution qexec = QueryExecutionFactory.create(query, model);
			LARQ.setDefaultIndex(qexec.getContext(), index);

			ResultSet rs = qexec.execSelect();
			try {
				while (rs.hasNext()) {
					QuerySolution qs = rs.next();

					result.add(qs.getResource("sp").getURI());
				}
			} finally {
				qexec.close();
			}
		} finally {
			dataset.end();
		}

		return result;
	}



	public boolean hasDomain(String relURI, String typeURI) {
		boolean result = false;
		dataset.begin(ReadWrite.READ);
		try {
			String queryString = prefixes + "ASK { <" + URIref.encode(relURI)
					+ "> rdfs:domain ?t. \n" + "<" + URIref.encode(typeURI)
					+ "> rdfs:subClassOf* ?t }";
			Query query = QueryFactory.create(queryString);
			QueryExecution qexec = QueryExecutionFactory.create(query, model);
			LARQ.setDefaultIndex(qexec.getContext(), index);

			result = qexec.execAsk();

			qexec.close();
		} finally {
			dataset.end();
		}

		return result;
	}

	public boolean hasRange(String relURI, String typeURI) {
		boolean result = false;
		dataset.begin(ReadWrite.READ);
		try {
			String queryString = prefixes + "ASK { <" + URIref.encode(relURI)
					+ "> rdfs:range ?t. \n" + "<" + URIref.encode(typeURI)
					+ "> rdfs:subClassOf* ?t }";
			Query query = QueryFactory.create(queryString);
			QueryExecution qexec = QueryExecutionFactory.create(query, model);
			LARQ.setDefaultIndex(qexec.getContext(), index);

			result = qexec.execAsk();

			qexec.close();
		} finally {
			dataset.end();
		}

		return result;
	}

	/**
	 * Tells whether two entities are related with a given relationship (whether
	 * the triple exists or not)
	 * 
	 * @param subjURI
	 * @param propertyURI
	 * @param objURI
	 * @return
	 */
	public boolean existsTriple(String subjURI, String propertyURI,
			String objURI) {
		boolean result;
		dataset.begin(ReadWrite.READ);
		try {
			String queryString = prefixes + "ASK { <" + URIref.encode(subjURI)
					+ "> <" + URIref.encode(propertyURI) + "> <"
					+ URIref.encode(objURI) + ">}";
			Query query = QueryFactory.create(queryString);
			QueryExecution qexec = QueryExecutionFactory.create(query, model);
			LARQ.setDefaultIndex(qexec.getContext(), index);
			result = qexec.execAsk();
			qexec.close();
		} finally {
			dataset.end();
		}

		return result;
	}

	public HashSet<Literal> getMatchingNodes2(String value, int maxMatches) {
		HashSet<Literal> literals = new HashSet<Literal>();

		String q = null;
		Query query = null;

		dataset.begin(ReadWrite.READ);
		try {
			q = "PREFIX pf: <http://jena.hpl.hp.com/ARQ/property#>\n"
					+ "SELECT DISTINCT ?lit	WHERE { ?lit pf:textMatch ('"
					+ value + "' " + maxMatches + ")}";

			query = QueryFactory.create(q);
			QueryExecution qexec = QueryExecutionFactory.create(query, model);
			LARQ.setDefaultIndex(qexec.getContext(), index);

			ResultSet rs = qexec.execSelect();

			try {
				while (rs.hasNext()) {
					literals.add(rs.next().getLiteral("lit"));
				}
			} finally {
				qexec.close();
			}
		} catch (Exception ex) {
			System.out.println(q);
			System.out.println();
			System.out.println(query);
			ex.printStackTrace();
		} finally {
			dataset.end();
		}

		return literals;
	}


}