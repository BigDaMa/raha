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

import com.google.common.base.Stopwatch;
import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import qa.qcri.nadeef.core.datamodel.*;
import qa.qcri.nadeef.core.utils.Bootstrap;
import qa.qcri.nadeef.core.utils.CSVTools;
import qa.qcri.nadeef.core.utils.sql.DBConnectionPool;
import qa.qcri.nadeef.core.utils.sql.SQLDialectBase;
import qa.qcri.nadeef.core.utils.sql.SQLDialectFactory;
import qa.qcri.nadeef.test.NadeefTestBase;
import qa.qcri.nadeef.test.TestDataRepository;
import qa.qcri.nadeef.tools.DBConfig;

import java.io.File;
import java.sql.Connection;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Set;
import java.util.concurrent.TimeUnit;

/**
 * SourceImport Test.
 */
@RunWith(Parameterized.class)
public class SQLTableTest extends NadeefTestBase {
    private String tableName;
    private String tableName10k;
    private DBConnectionPool connectionFactory;

    public SQLTableTest(String config_) {
        super(config_);
    }

    @Before
    public void setup() {
        try {
            Bootstrap.start(testConfig);
            DBConfig dbConfig = NadeefConfiguration.getDbConfig();
            SQLDialectBase dialectManager =
                SQLDialectFactory.getDialectManagerInstance(dbConfig.getDialect());
            tableName =
                CSVTools.dump(dbConfig, dialectManager, TestDataRepository.getDumpTestCSVFile());
            tableName10k =
                CSVTools.dump(
                    dbConfig,
                    dialectManager,
                    new File("test/src/qa/qcri/nadeef/test/input/hospital_10k.csv"));
            connectionFactory = DBConnectionPool.createDBConnectionPool(dbConfig, dbConfig);
        } catch (Exception e) {
            e.printStackTrace();
            Assert.fail(e.getMessage());
        }
    }

    @After
    public void teardown() {
        try (
            Connection conn = connectionFactory.getSourceConnection();
            Statement stat = conn.createStatement()
        ) {
            SQLDialectBase dialectManager =
                SQLDialectFactory.getNadeefDialectManagerInstance();
            stat.execute(dialectManager.dropTable(tableName));
            stat.execute(dialectManager.dropTable(tableName10k));
            conn.commit();
            Bootstrap.shutdown();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Test
    public void testProjection() {
        SQLTable collection = new SQLTable(tableName, connectionFactory);
        collection.project(new Column(tableName + ".c"));
        Assert.assertEquals(12, collection.size());
        Tuple tuple = collection.get(0);
        Set cellSets = tuple.getCells();
        Assert.assertEquals(1, cellSets.size());
        Cell cell = (Cell)cellSets.iterator().next();
        Assert.assertTrue(
            (tableName + ".c").equalsIgnoreCase(cell.getColumn().getFullColumnName())
        );
    }

    @Test
    public void testFilter() {
        SQLTable collection = new SQLTable(tableName, connectionFactory);
        collection.filter(Predicate.createEq(new Column(tableName, "c"), "c1"))
                .project(new Column(tableName, "c"));
        Assert.assertEquals(7, collection.size());
        Tuple tuple = collection.get(0);
        Set cellSets = tuple.getCells();
        Assert.assertEquals(1, cellSets.size());
        Cell cell = (Cell)cellSets.iterator().next();
        Assert.assertTrue(
            (tableName + ".c").equalsIgnoreCase(cell.getColumn().getFullColumnName()));
    }

    @Test
    public void testGroup() {
        SQLTable collection = new SQLTable(tableName, connectionFactory);
        Column targetColumn = new Column(tableName, "c");
        Collection<Table> result = collection.groupOn(targetColumn);
        Assert.assertEquals(3, result.size());
        for (Table t : result) {
            Tuple tuple = t.get(0);
            String value =(String) tuple.get("c");
            switch (value) {
                case "c1":
                    Assert.assertEquals(7, t.size());
                    break;
                case "c3":
                    Assert.assertEquals(1, t.size());
                    break;
                case "c2":
                    Assert.assertEquals(4, t.size());
                    break;
            }

        }
    }

    @Test public void testGroup2() {
        SQLTable table = new SQLTable(tableName10k, connectionFactory);
        Column column0 = new Column(tableName10k, "hospitalowner");
        Column column1 = new Column(tableName10k, "condition");
        List<Column> columns = new ArrayList<>();
        columns.add(column0);
        columns.add(column1);
        Collection<Table> blocks = table.groupOn(columns);
        System.out.println("Blocked " + blocks.size() + " groups.");
        for (Table x : blocks) {
            Tuple tuple = x.get(0);
            for (int i = 1; i < x.size(); i ++) {
                Tuple tuple1 = x.get(i);
                Assert.assertEquals(tuple.get(column0), tuple1.get(column0));
                Assert.assertEquals(tuple.get(column0), tuple1.get(column0));
            }
        }
    }

    @Test public void testGroup3() {
        SQLTable table = new SQLTable(tableName10k, connectionFactory);
        Column column0 = new Column(tableName10k, "hospitalowner");
        Column column1 = new Column(tableName10k, "condition");
        List<Column> columns = new ArrayList<>();
        columns.add(column0);
        columns.add(column1);
        Collection<Table> blocks = table.groupOnConstrained(columns);
        System.out.println("Blocked " + blocks.size() + " groups.");
        for (Table x : blocks) {
            Tuple tuple = x.get(0);
            for (int i = 1; i < x.size(); i ++) {
                Tuple tuple1 = x.get(i);
                Assert.assertEquals(tuple.get(column0), tuple1.get(column0));
                Assert.assertEquals(tuple.get(column0), tuple1.get(column0));
            }
        }
    }
    // @Test
    public void testSize() throws InterruptedException {
        Stopwatch stopwatch = Stopwatch.createStarted();
        SQLTable table = new SQLTable("TB_test60m", connectionFactory);
        table.get(0);

        long elapsedTime = stopwatch.elapsed(TimeUnit.MILLISECONDS);
        int mb = 1024*1024;

        //Getting the runtime reference from system
        Runtime runtime = Runtime.getRuntime();

        System.out.println("##### Heap utilization statistics [MB] #####");

        long usedMemory = (runtime.totalMemory() - runtime.freeMemory()) / mb;
        System.out.println("Used Memory:" + usedMemory);
        System.out.println("Free Memory:" + runtime.freeMemory() / mb);
        System.out.println("Total Memory:" + runtime.totalMemory() / mb);
        System.out.println("Max Memory:" + runtime.maxMemory() / mb);
        System.out.println("Load time: " + elapsedTime + " ms.");

        // memory usage should be less than 700 mb.
        Assert.assertTrue(usedMemory < 700);

        // load time should be less than 18000 ms
        Assert.assertTrue(elapsedTime < 18000);
    }
}
