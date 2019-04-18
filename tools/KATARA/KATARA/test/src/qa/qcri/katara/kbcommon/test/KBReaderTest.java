package qa.qcri.katara.kbcommon.test;

import org.junit.*;

import qa.qcri.katara.common.test.TestUtil;
import qa.qcri.katara.kbcommon.KBReader;
import qa.qcri.katara.kbcommon.KnowledgeDatabaseConfig;
import qa.qcri.katara.kbcommon.PatternDiscoveryException;
import qa.qcri.katara.kbcommon.util.StringPair;

import java.util.*;

/**
 * Test KBReader
 * @author x4chu
 *
 */
public class KBReaderTest {
	
	@BeforeClass
	public static void setUp() throws Exception {
		
		System.out.println("inside the setup");
		
		KnowledgeDatabaseConfig.setDataDirectoryBase(TestUtil.getKDBPath());
		
		System.out.println("The kd path is: " + KnowledgeDatabaseConfig.dataDirectoryBase);
		
	}
	
	
	@Test
	public  void KBReaderTestSteps() throws Exception{
		
		System.out.println("*****Step0: Starting  KBReader");
		
		KBReader reader = new KBReader();
		System.out.println("*****Step1: Done Init KBReader");
		
		
//		reader.runQuery();
		
		
		reader.close();
		System.out.println("*****Step X: closed Reader");
	}
	
	
//	@Test
//	public  void KBReaderTestSteps() throws Exception{
//		
//		System.out.println("*****Step0: Starting  KBReader");
//		
//		KBReader reader = new KBReader();
//		
//		System.out.println("*****Step1: Done Init KBReader");
//		
//		
//		Set<String> allRelationships = reader.getAllRelationships();
//		
//		System.out.println("*****Step2: Num Rels: " + allRelationships.size());
//		
//		Set<String> candidateTypes = null;
//		if (KnowledgeDatabaseConfig.maxMatches == -1)
//			candidateTypes = reader.getTypesOfEntitiesWithLabel("Beijing");
//		else
//			candidateTypes = reader.getTypesOfEntitiesWithLabel("Beijing",
//					KnowledgeDatabaseConfig.maxMatches);
//		
//		System.out.println("*****Step3: Getting types: " + candidateTypes.size());
//		
//		Set<String> candidateRels = null;
//		if (KnowledgeDatabaseConfig.maxMatches == -1)
//			candidateRels = reader.getDirectRelationShips("China","Beijing",false);
//		else
//			candidateRels = reader.getDirectRelationShips("China","Beijing",false,
//					KnowledgeDatabaseConfig.maxMatches);
//		
//		System.out.println("*****Step4: Getting rels: " + candidateRels.size());
//		
//		
//		String firstRel = candidateRels.iterator().next();
//		Set<StringPair> sps = reader.getSubjectObjectGivenRel(firstRel);
//		System.out.println("*****Step5: Getting all subjects and object given a rel " + sps.size());
//		
//		reader.close();
//		System.out.println("*****Step X: closed Reader");
//	}
}
