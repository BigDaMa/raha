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
import qa.qcri.nadeef.core.datamodel.NadeefConfiguration;
import qa.qcri.nadeef.core.datamodel.SQLTable;
import qa.qcri.nadeef.core.datamodel.Table;
import qa.qcri.nadeef.core.utils.Bootstrap;
import qa.qcri.nadeef.core.utils.CSVTools;
import qa.qcri.nadeef.core.utils.sql.DBConnectionPool;
import qa.qcri.nadeef.core.utils.sql.SQLDialectBase;
import qa.qcri.nadeef.core.utils.sql.SQLDialectFactory;
import qa.qcri.nadeef.test.NadeefTestBase;
import qa.qcri.nadeef.tools.DBConfig;
import qa.qcri.nadeef.tools.PerfReport;

import java.io.File;
import java.sql.Connection;
import java.sql.Statement;
import java.util.Collection;
import java.util.concurrent.TimeUnit;

@RunWith(Parameterized.class)
public class TablePerformanceBenchmark extends NadeefTestBase {
    private String tableName10k;
    private String tableName30k;
    private DBConnectionPool connectionFactory;

    public TablePerformanceBenchmark(String config_) {
        super(config_);
    }

    @Before
    public void setup() {
        try {
            Bootstrap.start(testConfig);
            DBConfig dbConfig = NadeefConfiguration.getDbConfig();
            SQLDialectBase dialectManager =
                SQLDialectFactory.getDialectManagerInstance(dbConfig.getDialect());
            tableName10k =
                CSVTools.dump(
                    dbConfig,
                    dialectManager,
                    new File("test/src/qa/qcri/nadeef/test/input/hospital_10k.csv")
                );
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
            stat.execute(dialectManager.dropTable(tableName10k));
            conn.commit();
            Bootstrap.shutdown();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Test
    public void testLoad() {
        Stopwatch watch = Stopwatch.createStarted();
        Long before = PerfReport.get(PerfReport.Metric.DBConnectionCount).get(0);
        SQLTable table = new SQLTable(tableName10k, connectionFactory);
        int size = table.size();
        long elapsedTime = watch.elapsed(TimeUnit.MILLISECONDS);
        System.out.println(
            String.format("Loading %s %d rows Elapsed time %d ms.",
                tableName10k,
                size,
                elapsedTime));
        Long after = PerfReport.get(PerfReport.Metric.DBConnectionCount).get(0);
        System.out.println("DB connection count " + (after - before));
        watch.stop();
        Assert.assertEquals(10000, size);
    }

    @Test
    public void testBlock() {
        Stopwatch watch = Stopwatch.createStarted();
        SQLTable table = new SQLTable(tableName10k, connectionFactory);
        Long before = PerfReport.get(PerfReport.Metric.DBConnectionCount).get(0);
        Collection<Table> sqlTableList = table.groupOn("zipcode");
        long elapsedTime = watch.elapsed(TimeUnit.MILLISECONDS);
        System.out.println(
            String.format("Blocking %s into %d blocks Elapsed time %d ms.",
                tableName10k,
                sqlTableList.size(),
                elapsedTime));
        watch.reset();
        watch.start();
        int count = 0;
        for (Table x : sqlTableList) {
            count += x.size();
        }
        Long after = PerfReport.get(PerfReport.Metric.DBConnectionCount).get(0);
        elapsedTime = watch.elapsed(TimeUnit.MILLISECONDS);
        System.out.println(
            String.format("Total size %d Elapsed time %d ms with %d connections",
                count,
                elapsedTime,
                (after - before)));
        watch.stop();
        Assert.assertEquals(10000, count);
    }
}
