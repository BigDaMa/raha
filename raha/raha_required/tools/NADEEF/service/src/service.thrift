/**
 * NADEEF service thrift definition.
 */
namespace java qa.qcri.nadeef.service.thrift

enum TNadeefExceptionType{
    UNKNOWN = 0,
    COMPILE_ERROR = 1,
    INVALID_RULE = 2,
}

enum TJobStatusType {
    WAITING = 0,
    RUNNING = 1,
    NOTAVAILABLE = 2
}

struct TJobStatus {
    1: string key,
    2: TJobStatusType status,
    3: i32 overallProgress,
    4: list<string> names,
    5: list<i32> progress
}

struct TRule {
    1: string name,
    2: string type,
    3: string code
}

exception TNadeefRemoteException {
    1:TNadeefExceptionType type,
    2:optional string message
}

service TNadeefService {
    /**
     * Generates UDF code for the rule.
     * @param rule Rule.
     * @param tableName target table name.
     * @return generated code.
     */
    string generate(1:TRule rule, 2: string tableName, 3: string dbname)
        throws (1:TNadeefRemoteException re),

    /**
     * Verify the given rule.
     * @param rule input rule.
     */
    bool verify(1: TRule rule) throws (1:TNadeefRemoteException re),

    /**
     * Detect with the given rule.
     * @param rule input rule.
     * @param table1 table 1 name.
     * @param table2 table 2 name.
     * @param outputdb output database name.
     * @return job key.
     */
    string detect(
        1: TRule rule,
        2: string table1,
        3: string table2,
        4: string outputdb = 'nadeefdb'
    ) throws (1:TNadeefRemoteException re),

    /**
     * Repair with the given rule.
     * @param rule input rule.
     * @param table1 table 1 name.
     * @param table2 table 2 name.
     * @param outputdb output database name.
     * @return job key.
     */
    string repair(
        1: TRule rule,
        2: string table1,
        3: string table2,
        4: string outputdb = 'nadeefdb'
    ) throws (1:TNadeefRemoteException re),

    /**
     * Gets status of a specific job.
     * @param rule input rule.
     * @return job status.
     */
    TJobStatus getJobStatus(1: string key),

    /**
     * Gets all job status.
     * @return all the job status.
     */
    list<TJobStatus> getAllJobStatus()
}