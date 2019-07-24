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

package qa.qcri.nadeef.lab.hc.test;

import com.google.gson.JsonObject;
import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import qa.qcri.nadeef.core.datamodel.CleanPlan;
import qa.qcri.nadeef.core.datamodel.CleanPlanJsonBuilder;
import qa.qcri.nadeef.core.datamodel.NadeefConfiguration;
import qa.qcri.nadeef.core.pipeline.CleanExecutor;
import qa.qcri.nadeef.core.pipeline.UpdateExecutor;
import qa.qcri.nadeef.core.utils.Bootstrap;
import qa.qcri.nadeef.core.utils.Violations;
import qa.qcri.nadeef.core.utils.sql.DBInstaller;
import qa.qcri.nadeef.lab.hc.HolisticCleaning;
import qa.qcri.nadeef.tools.DBConfig;

public class DCRepairTest {
    private DBConfig dbConfig;

    @Before
    public void setup() {
        try {
            Bootstrap.start();
            NadeefConfiguration.setMaxIterationNumber(2);
            NadeefConfiguration.setAlwaysOverride(true);
            NadeefConfiguration.setDecisionMakerClass(HolisticCleaning.class);
            dbConfig = NadeefConfiguration.getDbConfig();

            DBInstaller.uninstall(NadeefConfiguration.getDbConfig());
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
    public void cleanExecutorConstantDCTest(){
        try{
            JsonObject obj =
                new CleanPlanJsonBuilder()
                .csv("test/src/qa/qcri/nadeef/test/input/dumptest.csv")
                .type("dc")
                .value("not(t1.B!=b1)").build();

            CleanPlan cleanPlan = CleanPlan.create(obj, dbConfig).get(0);
            CleanExecutor executor = new CleanExecutor(cleanPlan);
            executor.detect();
            verifyViolationResult(5);
            executor.repair();

            UpdateExecutor updateExecutor = new UpdateExecutor(cleanPlan);
            updateExecutor.run();

            int updatedCount = updateExecutor.getUpdateCellCount();
            Assert.assertEquals(5, updatedCount);
        } catch (Exception e){
            e.printStackTrace();
            Assert.fail(e.getMessage());
        }
    }

    @Test
    public void cleanExecutorDCTest(){
        CleanExecutor executor = null;
        try{
            JsonObject obj =
                new CleanPlanJsonBuilder()
                    .csv("test/src/qa/qcri/nadeef/test/input/dumptest3.csv")
                    .type("dc")
                    .value("not(t1.A>t1.B&t1.B=2)").build();

            CleanPlan cleanPlan = CleanPlan.create(obj, dbConfig).get(0);
            executor = new CleanExecutor(cleanPlan);
            executor.detect();
            verifyViolationResult(6);
            executor.repair();

            UpdateExecutor updateExecutor = new UpdateExecutor(cleanPlan);
            updateExecutor.run();

            int updatedCount = updateExecutor.getUpdateCellCount();
            Assert.assertEquals(3, updatedCount);

        } catch (Exception e){
            e.printStackTrace();
            Assert.fail(e.getMessage());
        } finally {
            if (executor != null) {
                executor.shutdown();
            }
        }
    }

    @Test
    public void cleanExecutorDCTest2(){
        CleanExecutor executor = null;
        try{
            JsonObject obj =
                new CleanPlanJsonBuilder()
                    .csv("test/src/qa/qcri/nadeef/test/input/employee.csv")
                    .type("dc")
                    .value("not(t1.ID=t2.ManagerID&t1.Salary<t2.Salary)").build();

            CleanPlan cleanPlan = CleanPlan.create(obj, dbConfig).get(0);
            executor = new CleanExecutor(cleanPlan);
            executor.detect();
            verifyViolationResult(4);
            executor.repair();

            UpdateExecutor updateExecutor = new UpdateExecutor(cleanPlan);
            updateExecutor.run();

            int updatedCount = updateExecutor.getUpdateCellCount();
            Assert.assertEquals(1, updatedCount);

        } catch (Exception e){
            e.printStackTrace();
            Assert.fail(e.getMessage());
        } finally {
            if (executor != null) {
                executor.shutdown();
            }
        }
    }

    @Test
    public void cleanExecutorDCTest3(){
        CleanExecutor executor = null;
        try{
            JsonObject obj =
                new CleanPlanJsonBuilder()
                    .csv("test/src/qa/qcri/nadeef/test/input/employee.csv")
                    .type("dc")
                    .value("not(t1.ID=t2.ManagerID&t1.Tax<t2.Tax)").build();

            CleanPlan cleanPlan = CleanPlan.create(obj, dbConfig).get(0);
            executor = new CleanExecutor(cleanPlan);
            executor.detect();
            verifyViolationResult(4);
            executor.repair();

            UpdateExecutor updateExecutor = new UpdateExecutor(cleanPlan);
            updateExecutor.run();

            int updatedCount = updateExecutor.getUpdateCellCount();
            Assert.assertEquals(1, updatedCount);

        } catch (Exception e){
            e.printStackTrace();
            Assert.fail(e.getMessage());
        } finally {
            if (executor != null) {
                executor.shutdown();
            }
        }
    }

    @Test
    public void cleanExecutorDCTest4(){
        CleanExecutor executor = null;
        try{
            JsonObject obj =
                new CleanPlanJsonBuilder()
                    .csv("test/src/qa/qcri/nadeef/test/input/salary.csv")
                    .type("dc")
                    .value("not(t1.Salary>t2.Salary&t1.Tax<t2.Tax)").build();

            CleanPlan cleanPlan = CleanPlan.create(obj, dbConfig).get(0);
            executor = new CleanExecutor(cleanPlan);
            executor.detect();
            verifyViolationResult(172);
            executor.repair();

            UpdateExecutor updateExecutor = new UpdateExecutor(cleanPlan);
            updateExecutor.run();

            int updatedCount = updateExecutor.getUpdateCellCount();
            Assert.assertEquals(5, updatedCount);

        } catch (Exception e){
            e.printStackTrace();
            Assert.fail(e.getMessage());
        } finally {
            if (executor != null)
                executor.shutdown();
        }
    }

    private void verifyViolationResult(int expectRow)
        throws Exception {
        int rowCount = Violations.getViolationRowCount(NadeefConfiguration.getDbConfig());
        Assert.assertEquals(expectRow, rowCount);
    }
}
