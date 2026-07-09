import sys
import traceback
import logging

from markets.logger.logging import get_logger
# use the dedicated exception logger so exceptions go to logs/exception.log
logger = get_logger("markets.exception")

class Markets_Exception(Exception) :
    def __init__(self, error_message: Exception, error_details: sys) :
        super().__init__(str(error_message))
        self.log_data = self.get_log_dict(error_message, error_details)
        #self.error_message = str(error_message)
        self.error_message = Markets_Exception.get_detailed_error_message(error_message=error_message, error_detail=error_details)
        # automatically log the error when exception is created
        try:
            self.log_error()
        except Exception:
            # avoid raising during logging
            pass

    @staticmethod
    def get_log_dict(error_message: Exception, error_details: sys) -> dict :
        exc_type, _, exc_tb = error_details.exc_info()
        tb_info = traceback.extract_tb(exc_tb)[-1]

        return {
            "file": tb_info.filename,
            "function": tb_info.name,
            "line": tb_info.lineno,
            "code": tb_info.line.strip() if tb_info.line else "N/A",
            "error_type": exc_type.__name__,
            "message": str(error_message)
        }

    @staticmethod
    def get_detailed_error_message(error_message:Exception, error_detail:sys)->str :
        '''
        error_message: Exception object
        error_detail: object of sys module
        '''
        _,_,exec_tb = error_detail.exc_info() # exception type, exception value, traceback object
        exception_block_line_number = exec_tb.tb_frame.f_lineno
        try_block_line_number = exec_tb.tb_lineno
        file_name = exec_tb.tb_frame.f_code.co_filename
        detailed_message = f"""
        Error occured in script:
        [{file_name}] at
        try block line number: [{try_block_line_number}] and exception block line number:[{exception_block_line_number}]
        error message: [{error_message}]
        """
        # Removed print(detailed_message) to avoid double printing
        return detailed_message

    def log_error(self) :
        logger.error(self.log_data)

    def __str__(self) -> str :
        # fall back to Exception's default string (the original error message)
        return super().__str__()

    def __repr__(self) :
        return f"{self.__class__.__name__}({super().__str__()})"