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

import com.google.gson.JsonObject;
import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import qa.qcri.nadeef.core.datamodel.CleanPlan;
import qa.qcri.nadeef.core.datamodel.CleanPlanJsonBuilder;
import qa.qcri.nadeef.core.datamodel.NadeefConfiguration;
import qa.qcri.nadeef.core.pipeline.CleanExecutor;
import qa.qcri.nadeef.core.utils.Bootstrap;
import qa.qcri.nadeef.core.utils.Violations;
import qa.qcri.nadeef.core.utils.sql.DBInstaller;
import qa.qcri.nadeef.test.NadeefTestBase;
import qa.qcri.nadeef.test.TestDataRepository;

import java.util.ArrayList;
import java.util.List;

@RunWith(Parameterized.class)
public class ERDetectionTest extends NadeefTestBase {
    public ERDetectionTest(String testConfig_) {
        super(testConfig_);
    }

    @Before
    public void setup() {
        try {
            Bootstrap.start(testConfig);
            DBInstaller.uninstall(NadeefConfiguration.getDbConfig());
        } catch (Exception ex) {
            Assert.fail(ex.getMessage());
        }
    }

    @After
    public void tearDown() {
        Bootstrap.shutdown();
    }

    @Test
    public void erTest1() throws Exception {
        try {
            CleanPlan cleanPlan = TestDataRepository.getPlan("ERCleanPlan1.json").get(0);
            CleanExecutor executor = new CleanExecutor(cleanPlan);
            executor.detect();
            verifyViolationResult(51);
        } catch (Exception e){
            e.printStackTrace();
            Assert.fail(e.getMessage());
        }
    }

    @Test
    public void erTest2() throws Exception {
        try {
            JsonObject obj =
                new CleanPlanJsonBuilder()
                    .sqltype("postgres")
                    .url("localhost/unittest")
                    .username("tester")
                    .password("tester")
                    .table("tb_qcars")
                    .target("tb_qcars")
                    .type("udf")
                    .value("qa.qcri.nadeef.test.udf.DedupRule")
                    .build();

            CleanPlan cleanPlan = CleanPlan.create(obj, NadeefConfiguration.getDbConfig()).get(0);
            CleanExecutor executor = new CleanExecutor(cleanPlan);
            executor.detect();
            verifyViolationResult(25440);
        } catch (Exception e){
            e.printStackTrace();
            Assert.fail(e.getMessage());
        }
    }

    @Test
    public void erTest3() throws Exception {
        try {
            List<String> rule = new ArrayList<>();
            rule.add("EQ(tb_qcars.model, tb_qcars.model)=1");
            rule.add("ED(tb_qcars.contact_number, tb_qcars.contact_number)>0.8");
            rule.add("EQ(tb_qcars.brand_name, tb_qcars.brand_name)=1");

            JsonObject obj =
                new CleanPlanJsonBuilder()
                    .sqltype("postgres")
                    .url("localhost/unittest")
                    .username("tester")
                    .password("tester")
                    .table("tb_qcars")
                    .target("tb_qcars")
                    .type("er")
                    .value(rule)
                    .build();

            CleanPlan cleanPlan = CleanPlan.create(obj, NadeefConfiguration.getDbConfig()).get(0);
            CleanExecutor executor = new CleanExecutor(cleanPlan);
            executor.detect();
            verifyViolationResult(25440);
        } catch (Exception e){
            e.printStackTrace();
            Assert.fail(e.getMessage());
        }
    }

    private void verifyViolationResult(int expectRow)
        throws Exception {
        int rowCount = Violations.getViolationRowCount(NadeefConfiguration.getDbConfig());
        Assert.assertEquals(expectRow, rowCount);
    }
}