
(cl:in-package :asdf)

(defsystem "fusion_sensor-msg"
  :depends-on (:roslisp-msg-protocol :roslisp-utils :std_msgs-msg
)
  :components ((:file "_package")
    (:file "MyArray" :depends-on ("_package_MyArray"))
    (:file "_package_MyArray" :depends-on ("_package"))
  ))