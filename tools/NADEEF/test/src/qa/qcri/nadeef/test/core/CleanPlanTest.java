/*
 * QCRI, NADEEF LICENSE
 * NADEEF is an extensible, generalized and easy-to-deploy data cleaning platform built at QCRI.
 * NADEEF means "Clean" in Arabic
 *
 * Copyright (c) 2011-2013, Qatar Foundation for Education, Science and Community Development (on
 * behalf of Qatar Computing Research Institute) having its principle place of business in Doha,
 * Qatar with the registered address P.O box 5825 Doha, Qatar (hereinafter referred to as "QCRI")
 *
 * NADEEF has patent pending nevertheless the following is granted.
 * NADEEF is released under the terms of the MIT License, (http://opensource.org/licenses/MIT).
 */

package qa.qcri.nadeef.test.core;

import org.junit.*;
import org.junit.rules.ExpectedException;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import qa.qcri.nadeef.core.datamodel.CleanPlan;
import qa.qcri.nadeef.core.datamodel.NadeefConfiguration;
import qa.qcri.nadeef.core.datamodel.Rule;
import qa.qcri.nadeef.core.utils.Bootstrap;
import qa.qcri.nadeef.core.utils.CSVTools;
import qa.qcri.nadeef.core.utils.sql.SQLDialectFactory;
import qa.qcri.nadeef.test.NadeefTestBase;
import qa.qcri.nadeef.test.TestDataRepository;
import qa.qcri.nadeef.tools.DBConfig;
import qa.qcri.nadeef.tools.sql.SQLDialect;

import java.io.FileReader;
import java.util.List;

/**
 * CleanPlan unit test
 */
@RunWith(value = Parameterized.class)
public class CleanPlanTest extends NadeefTestBase {
    public CleanPlanTest(String testConfig_) {
        super(testConfig_);
    }

    private ExpectedException thrown = ExpectedException.none();

    @Before
    public void setup() {
        Assume.assumeTrue(testConfig.contains("derby"));
        try {
            Bootstrap.start(testConfig);
            CSVTools.dump(
                NadeefConfiguration.getDbConfig(),
                SQLDialectFactory.getNadeefDialectManagerInstance(),
                TestDataRepository.getLocationData1(),
                "LOCATION",
                true
            );
        } catch (Exception ex) {
            ex.printStackTrace();
            Assert.fail(ex.getMessage());
        }
    }

    @After
    public void teardown() {
        Bootstrap.shutdown();
    }

    @Test
    public void createFromJSONTest() {
        try {
            List<CleanPlan> cleanPlans =
                CleanPlan.create(
                    new FileReader(TestDataRepository.getTestFile1()),
                    NadeefConfiguration.getDbConfig()
                );

            Assert.assertEquals(1, cleanPlans.size());
            CleanPlan cleanPlan = cleanPlans.get(0);

            DBConfig source = cleanPlan.getSourceDBConfig();
            Assert.assertEquals("jdbc:derby:memory:nadeefdb;create=true", source.getUrl());
            Assert.assertEquals(SQLDialect.DERBYMEMORY, source.getDialect());
            Rule rule = cleanPlan.getRule();
            List<String> tableNames = rule.getTableNames();
            Assert.assertEquals(1, tableNames.size());
            Assert.assertEquals("LOCATION_COPY", tableNames.get(0));
        } catch (Exception ex) {
            ex.printStackTrace();
            Assert.fail(ex.getMessage());
        }
    }

    @Test
    public void createFromJSONTest2() {
        thrown.expect(IllegalArgumentException.class);
        try {
            CleanPlan.create(
                new FileReader(TestDataRepository.getFailurePlanFile1()),
                NadeefConfiguration.getDbConfig()
            );
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Test
    public void createFromJSONTest3() {
        thrown.expect(IllegalArgumentException.class);
        try {
            CleanPlan.create(
                new FileReader(TestDataRepository.getFailurePlanFile2()),
                NadeefConfiguration.getDbConfig()
            );
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
